#!/usr/bin/env python3
"""Evaluate generated style-transfer images with external metric implementations.

This script is designed for the current Image2LoRA inference layout and avoids
reimplementing standard metrics when installable/reference implementations are
available. Metrics that require content images are skipped when content images
are not supplied.

Example without content images:
    python scripts/evaluate.py --cases_dir image_in_ppt --output_dir outputs/eval

Example with a manifest that includes content images:
    python scripts/evaluate.py --manifest outputs/eval_manifest.jsonl --output_dir outputs/eval

Manifest JSONL fields:
    {
      "prompt": "...",
      "style_ref": "path/to/style.jpg",
      "generated": "path/to/output.png",
      "method": "image2lora",
      "content": "path/to/content.jpg"
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_GENERATED_NAMES = ("output.png", "baseline.png")
DEFAULT_CONTENT_NAMES = ("content.png", "content.jpg", "content.jpeg")
TORCHVISION_VGG19_IMAGENET1K_V1_FILENAME = "vgg19-dcbb9e9d.pth"
FID_INCEPTION_FILENAME = "pt_inception-2015-12-05-6726825d.pth"
FID_INCEPTION_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"

METRIC_CATALOG: dict[str, dict[str, Any]] = {
    "ssim": {
        "content_dependency": "hard",
        "meaning": "Content/reference structural similarity. In style-transfer evaluation this is output vs content.",
        "implementation": "skimage.metrics.structural_similarity",
        "install": "pip install scikit-image",
        "source": "https://scikit-image.org/docs/stable/api/skimage.metrics.html",
    },
    "lpips": {
        "content_dependency": "hard_for_content_fidelity",
        "meaning": "Learned perceptual distance. In the MD/ArtFID setting this is output vs content.",
        "implementation": "richzhang/PerceptualSimilarity package: lpips",
        "install": "pip install lpips",
        "source": "https://github.com/richzhang/PerceptualSimilarity",
    },
    "fid": {
        "content_dependency": "none",
        "meaning": "Dataset-level distribution distance. Here: generated images vs style reference images.",
        "implementation": "mseitzer/pytorch-fid CLI",
        "install": "pip install pytorch-fid",
        "source": "https://github.com/mseitzer/pytorch-fid",
    },
    "artfid": {
        "content_dependency": "hard",
        "meaning": "Overall style-transfer score combining content fidelity and style distribution quality.",
        "implementation": "matthias-wright/art-fid CLI",
        "install": "pip install art-fid",
        "source": "https://github.com/matthias-wright/art-fid",
    },
    "style_loss": {
        "content_dependency": "none",
        "meaning": "VGG style-statistics distance between output and style reference.",
        "implementation": "Implemented here from the paper definition because no standalone standard metric package was found.",
        "install": "pip install torch torchvision pillow",
        "source": [
            "https://github.com/jcjohnson/fast-neural-style",
            "https://github.com/xunhuang1995/AdaIN-style",
        ],
        "variants": {
            "gram": "Gatys/Johnson-style Gram matrix MSE over VGG relu1_1..relu5_1.",
            "adain": "AdaIN-style VGG channel mean/std MSE over the same layers.",
        },
    },
}


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    style_ref: Path
    generated: Path
    method: str
    content: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate style-transfer outputs using existing metric implementations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cases_dir", type=Path, help="Directory containing one subdir per prompt/case.")
    input_group.add_argument("--manifest", type=Path, help="JSONL manifest with generated/style/content paths.")
    parser.add_argument(
        "--generated_names",
        nargs="+",
        default=list(DEFAULT_GENERATED_NAMES),
        help="Generated filenames to scan inside each case directory.",
    )
    parser.add_argument(
        "--content_names",
        nargs="+",
        default=list(DEFAULT_CONTENT_NAMES),
        help="Optional content filenames to scan inside each case directory.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval"), help="Report output directory.")
    parser.add_argument("--device", default="cuda", help="Device string forwarded to metric packages.")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    parser.add_argument("--skip_fid", action="store_true", help="Skip pytorch-fid.")
    parser.add_argument(
        "--fid_weights",
        type=Path,
        default=None,
        help="Optional local pytorch-fid Inception weights. Avoids network download.",
    )
    parser.add_argument("--skip_style_loss", action="store_true", help="Skip VGG style loss.")
    parser.add_argument("--skip_content_metrics", action="store_true", help="Skip SSIM, LPIPS, and ArtFID.")
    parser.add_argument("--style_loss_image_size", type=int, default=256, help="Resize/crop size for VGG style loss.")
    parser.add_argument(
        "--style_loss_vgg_weights",
        type=Path,
        default=None,
        help="Optional local torchvision VGG19 state_dict for style loss.",
    )
    parser.add_argument(
        "--copy_inputs",
        action="store_true",
        help="Copy images into metric staging dirs instead of symlinking.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path | None = None) -> Path:
    if path.is_absolute():
        return path
    if base is not None:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def load_cases_from_manifest(manifest: Path) -> list[EvalCase]:
    manifest = resolve_path(manifest)
    cases: list[EvalCase] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                style_ref = resolve_path(Path(record["style_ref"]), manifest.parent)
                generated = resolve_path(Path(record["generated"]), manifest.parent)
            except KeyError as exc:
                raise ValueError(f"Missing required field {exc} at {manifest}:{line_no}") from exc
            content = record.get("content") or record.get("content_image")
            cases.append(
                EvalCase(
                    prompt=str(record.get("prompt") or generated.parent.name),
                    style_ref=style_ref,
                    generated=generated,
                    method=str(record.get("method") or generated.stem),
                    content=resolve_path(Path(content), manifest.parent) if content else None,
                )
            )
    return cases


def infer_style_ref(case_dir: Path, generated_names: set[str], content_names: set[str]) -> Path | None:
    for path in sorted(case_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name in generated_names or path.name in content_names:
            continue
        return path
    return None


def infer_content(case_dir: Path, content_names: list[str]) -> Path | None:
    for name in content_names:
        path = case_dir / name
        if path.exists():
            return path
    return None


def load_cases_from_dir(cases_dir: Path, generated_names: list[str], content_names: list[str]) -> list[EvalCase]:
    cases_dir = resolve_path(cases_dir)
    generated_name_set = set(generated_names)
    content_name_set = set(content_names)
    cases: list[EvalCase] = []
    for case_dir in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
        style_ref = infer_style_ref(case_dir, generated_name_set, content_name_set)
        if style_ref is None:
            print(f"[warn] Skip {case_dir}: no style reference image found.", file=sys.stderr)
            continue
        content = infer_content(case_dir, content_names)
        for generated_name in generated_names:
            generated = case_dir / generated_name
            if generated.exists():
                cases.append(
                    EvalCase(
                        prompt=case_dir.name,
                        style_ref=style_ref,
                        generated=generated,
                        method=generated.stem,
                        content=content,
                    )
                )
    return cases


def validate_cases(cases: list[EvalCase]) -> list[EvalCase]:
    valid = []
    for case in cases:
        required = [case.style_ref, case.generated]
        missing = [str(path) for path in required if not path.exists()]
        if case.content is not None and not case.content.exists():
            missing.append(str(case.content))
        if missing:
            print(f"[warn] Skip {case.generated}: missing {missing}", file=sys.stderr)
            continue
        valid.append(case)
    if not valid:
        raise ValueError("No valid evaluation cases found.")
    return valid


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_inputs: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_inputs:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_metric_dirs(cases: list[EvalCase], output_dir: Path, copy_inputs: bool) -> dict[str, Any]:
    stage_root = output_dir / "_metric_inputs"
    reset_dir(stage_root)
    style_dir = stage_root / "style_refs"
    style_dir.mkdir(parents=True, exist_ok=True)

    seen_style: dict[Path, Path] = {}
    for idx, style_ref in enumerate(sorted({case.style_ref for case in cases})):
        dst = style_dir / f"{idx:06d}{style_ref.suffix.lower()}"
        link_or_copy(style_ref, dst, copy_inputs)
        seen_style[style_ref] = dst

    by_method: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_method[case.method].append(case)

    method_dirs: dict[str, dict[str, Path]] = {}
    for method, method_cases in sorted(by_method.items()):
        gen_dir = stage_root / f"generated_{method}"
        content_dir = stage_root / f"content_{method}"
        stylized_for_artfid_dir = stage_root / f"stylized_{method}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        has_content = all(case.content is not None for case in method_cases)
        if has_content:
            content_dir.mkdir(parents=True, exist_ok=True)
            stylized_for_artfid_dir.mkdir(parents=True, exist_ok=True)

        for idx, case in enumerate(method_cases):
            gen_dst = gen_dir / f"{idx:06d}{case.generated.suffix.lower()}"
            link_or_copy(case.generated, gen_dst, copy_inputs)
            if has_content and case.content is not None:
                suffix = case.content.suffix.lower()
                content_dst = content_dir / f"{idx:06d}{suffix}"
                stylized_dst = stylized_for_artfid_dir / f"{idx:06d}{case.generated.suffix.lower()}"
                link_or_copy(case.content, content_dst, copy_inputs)
                link_or_copy(case.generated, stylized_dst, copy_inputs)

        method_dirs[method] = {"generated": gen_dir}
        if has_content:
            method_dirs[method]["content"] = content_dir
            method_dirs[method]["stylized_for_artfid"] = stylized_for_artfid_dir

    return {"stage_root": stage_root, "style": style_dir, "methods": method_dirs}


def run_command(command: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout


def parse_last_float(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return float(matches[-1]) if matches else None


def default_torch_checkpoint_path() -> Path:
    return (
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        / "hub"
        / "checkpoints"
        / FID_INCEPTION_FILENAME
    )


def prepare_fid_env(fid_weights: Path | None) -> tuple[dict[str, str] | None, str | None]:
    if fid_weights is None:
        cache_path = default_torch_checkpoint_path()
        if cache_path.exists():
            return None, None
        return (
            None,
            f"Missing cached FID Inception weights: {cache_path}. "
            f"Download {FID_INCEPTION_URL} to that path, or pass --fid_weights /path/to/{FID_INCEPTION_FILENAME}.",
        )

    weights_path = resolve_path(fid_weights)
    if not weights_path.exists():
        return None, f"Missing --fid_weights file: {weights_path}"

    fid_torch_home = Path(tempfile.gettempdir()) / "imagelora_fid_torch_home"
    checkpoint_dir = fid_torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cached_file = checkpoint_dir / FID_INCEPTION_FILENAME
    if cached_file.exists() or cached_file.is_symlink():
        cached_file.unlink()
    try:
        os.symlink(weights_path, cached_file)
    except OSError:
        shutil.copy2(weights_path, cached_file)

    env = os.environ.copy()
    env["TORCH_HOME"] = str(fid_torch_home)
    return env, None


def compute_fid(
    style_dir: Path,
    method_dirs: dict[str, dict[str, Path]],
    device: str,
    fid_weights: Path | None = None,
) -> dict[str, dict[str, Any]]:
    results = {}
    fid_env, skip_reason = prepare_fid_env(fid_weights)
    for method, dirs in sorted(method_dirs.items()):
        command = [sys.executable, "-m", "pytorch_fid", str(dirs["generated"]), str(style_dir), "--device", device]
        if skip_reason is not None:
            results[method] = {
                "value": None,
                "status": "skipped",
                "command": " ".join(command),
                "message": skip_reason,
                "weights_url": FID_INCEPTION_URL,
            }
            continue

        code, output = run_command(command, env=fid_env)
        value = parse_last_float(output) if code == 0 else None
        results[method] = {
            "value": value,
            "status": "ok" if value is not None else "skipped",
            "command": " ".join(command),
            "message": output.strip(),
        }
        if code != 0:
            results[method]["install"] = METRIC_CATALOG["fid"]["install"]
    return results


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def compute_ssim(cases: list[EvalCase]) -> dict[str, Any]:
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc), "install": METRIC_CATALOG["ssim"]["install"]}

    rows = []
    for case in cases:
        if case.content is None:
            continue
        content = load_rgb(case.content)
        generated = load_rgb(case.generated, size=content.size)
        value = structural_similarity(np.asarray(content), np.asarray(generated), channel_axis=2, data_range=255)
        rows.append({"generated": str(case.generated), "method": case.method, "value": float(value)})
    return {"status": "ok" if rows else "skipped", "rows": rows, "reason": None if rows else "no content images"}


def image_to_lpips_tensor(path: Path, size: tuple[int, int] | None = None) -> Any:
    import numpy as np
    import torch

    image = load_rgb(path, size=size)
    array = np.asarray(image).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def compute_lpips(cases: list[EvalCase], net: str, device: str) -> dict[str, Any]:
    try:
        import lpips
        import torch
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc), "install": METRIC_CATALOG["lpips"]["install"]}

    torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(torch_device).eval()
    rows = []
    with torch.no_grad():
        for case in cases:
            if case.content is None:
                continue
            content_image = load_rgb(case.content)
            content = image_to_lpips_tensor(case.content).to(torch_device)
            generated = image_to_lpips_tensor(case.generated, size=content_image.size).to(torch_device)
            value = loss_fn(content, generated).mean().item()
            rows.append({"generated": str(case.generated), "method": case.method, "value": float(value)})
    return {"status": "ok" if rows else "skipped", "rows": rows, "reason": None if rows else "no content images"}


def compute_artfid(style_dir: Path, method_dirs: dict[str, dict[str, Path]], device: str) -> dict[str, dict[str, Any]]:
    results = {}
    for method, dirs in sorted(method_dirs.items()):
        if "content" not in dirs:
            results[method] = {"status": "skipped", "reason": "no content images"}
            continue
        command = [
            sys.executable,
            "-m",
            "art_fid",
            "--style_images",
            str(style_dir),
            "--content_images",
            str(dirs["content"]),
            "--stylized_images",
            str(dirs["stylized_for_artfid"]),
            "--device",
            "cuda" if device.startswith("cuda") else "cpu",
        ]
        code, output = run_command(command)
        results[method] = {
            "value": parse_last_float(output) if code == 0 else None,
            "status": "ok" if code == 0 else "skipped",
            "command": " ".join(command),
            "message": output.strip(),
        }
        if code != 0:
            results[method]["install"] = METRIC_CATALOG["artfid"]["install"]
    return results


def metric_torch_device(device: str) -> Any:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def load_style_loss_tensor(path: Path, image_size: int, device: Any) -> Any:
    import numpy as np
    import torch

    image = load_rgb(path)
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    cropped = resized.crop((left, top, left + image_size, top + image_size))
    array = np.asarray(cropped).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_vgg19_features(weights_path: Path | None, device: Any) -> Any:
    import torch
    from torchvision import models

    if weights_path is not None:
        model = models.vgg19(weights=None)
        model.load_state_dict(torch.load(resolve_path(weights_path), map_location="cpu"))
    else:
        cache_path = (
            Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
            / "hub"
            / "checkpoints"
            / TORCHVISION_VGG19_IMAGENET1K_V1_FILENAME
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached VGG19 weights: {cache_path}. "
                "Pass --style_loss_vgg_weights or prepare the checkpoint as described in README_evaluate.md."
            )
        model = models.vgg19(weights=None)
        model.load_state_dict(torch.load(cache_path, map_location="cpu"))
    return model.features.to(device).eval()


def extract_vgg_style_features(vgg_features: Any, tensor: Any) -> list[Any]:
    layer_ids = {1, 6, 11, 20, 29}
    features = []
    h = tensor
    for idx, layer in enumerate(vgg_features):
        h = layer(h)
        if idx in layer_ids:
            features.append(h)
        if idx >= max(layer_ids):
            break
    return features


def gram_matrix(feature: Any) -> Any:
    b, c, h, w = feature.shape
    flat = feature.reshape(b, c, h * w)
    return flat.bmm(flat.transpose(1, 2)) / float(c * h * w)


def style_loss_values(output_features: list[Any], style_features: list[Any]) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    gram_losses = []
    adain_losses = []
    for output_feature, style_feature in zip(output_features, style_features):
        gram_losses.append(F.mse_loss(gram_matrix(output_feature), gram_matrix(style_feature)))

        output_mean = output_feature.mean(dim=(2, 3))
        style_mean = style_feature.mean(dim=(2, 3))
        output_std = output_feature.std(dim=(2, 3), unbiased=False)
        style_std = style_feature.std(dim=(2, 3), unbiased=False)
        adain_losses.append(F.mse_loss(output_mean, style_mean) + F.mse_loss(output_std, style_std))
    return float(torch.stack(gram_losses).sum().item()), float(torch.stack(adain_losses).sum().item())


def compute_style_loss(cases: list[EvalCase], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc), "install": METRIC_CATALOG["style_loss"]["install"]}

    try:
        device = metric_torch_device(args.device)
        vgg_features = load_vgg19_features(args.style_loss_vgg_weights, device)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "install": METRIC_CATALOG["style_loss"]["install"],
            "source": METRIC_CATALOG["style_loss"]["source"],
        }

    rows = []
    style_feature_cache = {}
    with torch.no_grad():
        for case in cases:
            if case.style_ref not in style_feature_cache:
                style_tensor = load_style_loss_tensor(case.style_ref, args.style_loss_image_size, device)
                style_feature_cache[case.style_ref] = extract_vgg_style_features(vgg_features, style_tensor)
            output_tensor = load_style_loss_tensor(case.generated, args.style_loss_image_size, device)
            output_features = extract_vgg_style_features(vgg_features, output_tensor)
            gram_loss, adain_loss = style_loss_values(output_features, style_feature_cache[case.style_ref])
            rows.append(
                {
                    "prompt": case.prompt,
                    "method": case.method,
                    "generated": str(case.generated),
                    "style_ref": str(case.style_ref),
                    "style_loss_gram": gram_loss,
                    "style_loss_adain": adain_loss,
                }
            )

    return {
        "status": "ok",
        "rows": rows,
        "image_size": args.style_loss_image_size,
        "layers": ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        "lower_is_better": True,
    }


def summarize_pairwise(metric_result: dict[str, Any]) -> dict[str, Any]:
    rows = metric_result.get("rows") or []
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(float(row["value"]))
    summary = {}
    for method, values in sorted(by_method.items()):
        summary[method] = {
            "mean": sum(values) / len(values),
            "num_images": len(values),
            "values": values,
        }
    return summary


def summarize_style_loss(metric_result: dict[str, Any]) -> dict[str, Any]:
    rows = metric_result.get("rows") or []
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(
            {
                "style_loss_gram": float(row["style_loss_gram"]),
                "style_loss_adain": float(row["style_loss_adain"]),
            }
        )
    summary = {}
    for method, values in sorted(by_method.items()):
        summary[method] = {
            "num_images": len(values),
            "style_loss_gram_mean": sum(v["style_loss_gram"] for v in values) / len(values),
            "style_loss_adain_mean": sum(v["style_loss_adain"] for v in values) / len(values),
        }
    return summary


def write_pairwise_csv(metric_name: str, metric_result: dict[str, Any], output_dir: Path) -> None:
    rows = metric_result.get("rows") or []
    if not rows:
        return
    path = output_dir / f"{metric_name}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "generated", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_style_loss_csv(metric_result: dict[str, Any], output_dir: Path) -> None:
    rows = metric_result.get("rows") or []
    if not rows:
        return
    path = output_dir / "style_loss.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "method",
                "generated",
                "style_ref",
                "style_loss_gram",
                "style_loss_adain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases_from_dir(args.cases_dir, args.generated_names, args.content_names) if args.cases_dir else load_cases_from_manifest(args.manifest)
    cases = validate_cases(cases)
    staged = stage_metric_dirs(cases, output_dir, args.copy_inputs)

    has_any_content = any(case.content is not None for case in cases)
    report: dict[str, Any] = {
        "num_cases": len(cases),
        "has_any_content_images": has_any_content,
        "metric_catalog": METRIC_CATALOG,
        "metrics": {},
    }

    if args.skip_fid:
        report["metrics"]["fid"] = {"status": "skipped", "reason": "--skip_fid"}
    else:
        report["metrics"]["fid"] = compute_fid(staged["style"], staged["methods"], args.device, args.fid_weights)

    if args.skip_content_metrics:
        for metric_name in ("ssim", "lpips", "artfid"):
            report["metrics"][metric_name] = {"status": "skipped", "reason": "--skip_content_metrics"}
    elif not has_any_content:
        for metric_name in ("ssim", "lpips", "artfid"):
            report["metrics"][metric_name] = {
                "status": "skipped",
                "reason": "content images are required by this metric under the standard style-transfer definition",
            }
    else:
        ssim_result = compute_ssim(cases)
        lpips_result = compute_lpips(cases, args.lpips_net, args.device)
        report["metrics"]["ssim"] = {**ssim_result, "summary": summarize_pairwise(ssim_result)}
        report["metrics"]["lpips"] = {**lpips_result, "summary": summarize_pairwise(lpips_result)}
        report["metrics"]["artfid"] = compute_artfid(staged["style"], staged["methods"], args.device)
        write_pairwise_csv("ssim", ssim_result, output_dir)
        write_pairwise_csv("lpips", lpips_result, output_dir)

    if args.skip_style_loss:
        report["metrics"]["style_loss"] = {"status": "skipped", "reason": "--skip_style_loss"}
    else:
        style_loss_result = compute_style_loss(cases, args)
        report["metrics"]["style_loss"] = {
            **style_loss_result,
            "summary": summarize_style_loss(style_loss_result),
            "source": METRIC_CATALOG["style_loss"]["source"],
        }
        write_style_loss_csv(style_loss_result, output_dir)

    return report


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    output_dir = resolve_path(args.output_dir)
    report_path = output_dir / "metrics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved metric report to {report_path}")


if __name__ == "__main__":
    main()
