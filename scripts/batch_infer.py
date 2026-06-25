#!/usr/bin/env python3
"""Batch inference for content/style image folders.

Content image filename format:
    type_caption.ext

Style reference filename format:
    style____*.jpg

Generated filename format:
    content.ext&&style.ext.jpg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ContentImage:
    path: Path
    content_type: str
    caption: str


@dataclass(frozen=True)
class StyleImage:
    path: Path
    style: str


@dataclass(frozen=True)
class BatchCase:
    content_image: str
    content_type: str
    caption: str
    prompt: str
    style_image: str
    style: str
    output_image: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Image2LoRA inference for all content/style pairs in folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_argument_group("inputs")
    input_group.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Folder containing both content images and style reference images.",
    )
    input_group.add_argument(
        "--content_dir",
        type=Path,
        default=None,
        help="Folder containing content images. Defaults to --input_dir.",
    )
    input_group.add_argument(
        "--style_dir",
        type=Path,
        default=None,
        help="Folder containing style reference images. Defaults to --input_dir.",
    )
    input_group.add_argument("--recursive", action="store_true", help="Search input folders recursively.")

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--output_dir", type=Path, required=True, help="Folder for generated images.")
    output_group.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output JSON manifest. Defaults to <output_dir>/manifest.json.",
    )
    output_group.add_argument("--skip_existing", action="store_true", help="Do not rerun cases with existing outputs.")
    output_group.add_argument("--dry_run", action="store_true", help="Parse and write manifest without running infer.")
    output_group.add_argument("--continue_on_error", action="store_true", help="Continue after an infer subprocess fails.")

    infer_group = parser.add_argument_group("infer options")
    infer_group.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    infer_group.add_argument("--checkpoint_dir", type=str, required=True)
    infer_group.add_argument(
        "--prompt_template",
        type=str,
        default="{caption}",
        help="Template for infer prompt. Available fields: {type}, {caption}, {style}, {content_filename}, {style_filename}.",
    )
    infer_group.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    infer_group.add_argument("--num_inference_steps", type=int, default=30)
    infer_group.add_argument("--guidance_scale", type=float, default=7.5)
    infer_group.add_argument("--seed", type=int, default=42)
    infer_group.add_argument("--down_dim", type=int, default=64)
    infer_group.add_argument("--up_dim", type=int, default=32)
    infer_group.add_argument("--rank", type=int, default=1)
    infer_group.add_argument("--decoder_blocks", type=int, default=8)
    infer_group.add_argument("--sample_iters", type=int, default=4)
    infer_group.add_argument("--resolution", type=int, default=512)
    infer_group.add_argument("--lora_scale", type=float, default=1.0)
    infer_group.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def iter_image_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and "&&" not in path.name
    )


def parse_style(path: Path) -> StyleImage | None:
    if "____" not in path.stem:
        return None
    style, _rest = path.stem.split("____", 1)
    if not style:
        return None
    return StyleImage(path=path, style=style)


def parse_content(path: Path) -> ContentImage | None:
    if parse_style(path) is not None:
        return None
    if "_" not in path.stem:
        return None
    content_type, caption = path.stem.split("_", 1)
    if not content_type or not caption:
        return None
    return ContentImage(path=path, content_type=content_type, caption=caption)


def discover_inputs(content_dir: Path, style_dir: Path, recursive: bool) -> tuple[list[ContentImage], list[StyleImage]]:
    contents = [item for path in iter_image_files(content_dir, recursive) if (item := parse_content(path)) is not None]
    styles = [item for path in iter_image_files(style_dir, recursive) if (item := parse_style(path)) is not None]
    if not contents:
        raise ValueError(f"No content images found in {content_dir}. Expected filenames like type_caption.ext")
    if not styles:
        raise ValueError(f"No style images found in {style_dir}. Expected filenames like style____*.jpg")
    return contents, styles


def make_prompt(template: str, content: ContentImage, style: StyleImage) -> str:
    return template.format(
        type=content.content_type,
        caption=content.caption,
        style=style.style,
        content_filename=content.path.name,
        style_filename=style.path.name,
    )


def build_infer_command(args: argparse.Namespace, style: StyleImage, prompt: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "infer.py"),
        "--pretrained_model",
        args.pretrained_model,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--ref_image",
        str(style.path),
        "--prompt",
        prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--output",
        str(output),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--down_dim",
        str(args.down_dim),
        "--up_dim",
        str(args.up_dim),
        "--rank",
        str(args.rank),
        "--decoder_blocks",
        str(args.decoder_blocks),
        "--sample_iters",
        str(args.sample_iters),
        "--resolution",
        str(args.resolution),
        "--lora_scale",
        str(args.lora_scale),
        "--dinov2_model_path",
        args.dinov2_model_path,
    ]


def write_manifest(path: Path, records: list[BatchCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.input_dir is None and (args.content_dir is None or args.style_dir is None):
        raise ValueError("Pass --input_dir, or pass both --content_dir and --style_dir.")

    input_dir = resolve_path(args.input_dir) if args.input_dir is not None else None
    content_dir = resolve_path(args.content_dir) if args.content_dir is not None else input_dir
    style_dir = resolve_path(args.style_dir) if args.style_dir is not None else input_dir
    if content_dir is None or style_dir is None:
        raise ValueError("Could not resolve content/style folders.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = resolve_path(args.manifest) if args.manifest is not None else output_dir / "manifest.json"

    contents, styles = discover_inputs(content_dir, style_dir, args.recursive)
    records: list[BatchCase] = []
    total = len(contents) * len(styles)
    case_idx = 0

    for style in styles:
        print(f"Style {style.style}: generating {len(contents)} caption images", flush=True)
        for content in contents:
            case_idx += 1
            prompt = make_prompt(args.prompt_template, content, style)
            output = output_dir / f"{content.path.name}&&{style.path.name}.jpg"
            status = "dry_run" if args.dry_run else "generated"

            if args.skip_existing and output.exists():
                status = "skipped_existing"
            elif not args.dry_run:
                print(f"[{case_idx}/{total}] {content.path.name} + {style.path.name}", flush=True)
                command = build_infer_command(args, style, prompt, output)
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0:
                    status = f"failed:{completed.returncode}"
                    records.append(
                        BatchCase(
                            content_image=str(content.path),
                            content_type=content.content_type,
                            caption=content.caption,
                            prompt=prompt,
                            style_image=str(style.path),
                            style=style.style,
                            output_image=str(output),
                            status=status,
                        )
                    )
                    write_manifest(manifest, records)
                    if not args.continue_on_error:
                        raise subprocess.CalledProcessError(completed.returncode, command)
                    continue

            records.append(
                BatchCase(
                    content_image=str(content.path),
                    content_type=content.content_type,
                    caption=content.caption,
                    prompt=prompt,
                    style_image=str(style.path),
                    style=style.style,
                    output_image=str(output),
                    status=status,
                )
            )
            write_manifest(manifest, records)

    print(f"Wrote {len(records)} records to {manifest}")


if __name__ == "__main__":
    main()
