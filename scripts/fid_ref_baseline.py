#!/usr/bin/env python3
"""Reference-vs-reference FID baseline by style.

Input files are expected to look like:
    content.ext&&style____*.ext

For each style, all reference images are shuffled and split into two groups.
The script then computes FID between the two groups using the same FID helper
used by scripts/evaluate.py and scripts/batch_eval.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import compute_fid, link_or_copy, reset_dir  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a reference-vs-reference FID baseline for each style.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fid_style_ref_dir",
        type=Path,
        required=True,
        help="Folder containing files named content.ext&&style____*.ext.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for baseline report.")
    parser.add_argument("--recursive", action="store_true", help="Search reference folder recursively.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for style splits.")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of random splits per style.")
    parser.add_argument(
        "--include_self_check",
        action="store_true",
        help="Also compute FID for each style reference set against itself. This should be near 0.",
    )
    parser.add_argument("--device", default="cuda", help="Device string forwarded to pytorch-fid.")
    parser.add_argument(
        "--fid_weights",
        type=Path,
        default=None,
        help="Optional local pytorch-fid Inception weights. Avoids network download.",
    )
    parser.add_argument(
        "--fid_min_images",
        type=int,
        default=2,
        help="Skip FID when either split has fewer images than this.",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=1,
        help="Batch size for pytorch-fid. Use 1 when image sizes differ.",
    )
    parser.add_argument(
        "--copy_inputs",
        action="store_true",
        help="Copy images into metric staging dirs instead of symlinking.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def iter_image_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in folder.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def parse_style_from_ref_name(path: Path) -> str | None:
    if "&&" not in path.name:
        return None
    _content_name, style_name = path.name.split("&&", 1)
    if "____" not in style_name:
        return None
    style, _rest = style_name.split("____", 1)
    return style or None


def load_refs_by_style(ref_dir: Path, recursive: bool) -> dict[str, list[Path]]:
    refs_by_style: dict[str, list[Path]] = defaultdict(list)
    for path in iter_image_files(ref_dir, recursive):
        style = parse_style_from_ref_name(path)
        if style is not None:
            refs_by_style[style].append(path)
    if not refs_by_style:
        raise ValueError(f"No reference images found in {ref_dir}. Expected filenames like content.ext&&style____*.ext")
    return {style: sorted(paths) for style, paths in refs_by_style.items()}


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def split_refs(style: str, refs: list[Path], seed: int, split_idx: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(refs)
    rng = random.Random(f"{seed}:{split_idx}:{style}")
    rng.shuffle(shuffled)
    split_at = len(shuffled) // 2
    return shuffled[:split_at], shuffled[split_at:]


def stage_split_dirs(
    style: str,
    split_idx: int,
    group_a: list[Path],
    group_b: list[Path],
    output_dir: Path,
    copy_inputs: bool,
) -> dict[str, Any]:
    split_name = "self_check" if split_idx < 0 else f"split_{split_idx:03d}"
    split_dir = output_dir / "by_style" / safe_name(style) / split_name
    stage_root = split_dir / "_metric_inputs"
    reset_dir(stage_root)
    group_a_dir = stage_root / "reference_group_a"
    group_b_dir = stage_root / "reference_group_b"
    group_a_dir.mkdir(parents=True, exist_ok=True)
    group_b_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(group_a):
        link_or_copy(path, group_a_dir / f"{idx:06d}{path.suffix.lower()}", copy_inputs)
    for idx, path in enumerate(group_b):
        link_or_copy(path, group_b_dir / f"{idx:06d}{path.suffix.lower()}", copy_inputs)

    return {
        "split_dir": split_dir,
        "group_a": group_a_dir,
        "group_b": group_b_dir,
        "methods": {style: {"generated": group_a_dir}},
    }


def compute_reference_self_check(style: str, refs: list[Path], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    staged = stage_split_dirs(style, -1, refs, refs, output_dir, args.copy_inputs)
    fid = compute_fid(
        staged["group_b"],
        staged["methods"],
        args.device,
        args.fid_weights,
        args.fid_min_images,
        args.fid_batch_size,
    )
    result = {
        "num_refs": len(refs),
        "fid": fid,
    }
    with (staged["split_dir"] / "metrics_report.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def summarize_style_trials(trials: list[dict[str, Any]], style: str) -> dict[str, Any]:
    values = []
    for trial in trials:
        fid = trial["fid"].get(style, {})
        if fid.get("status") == "ok" and fid.get("value") is not None:
            values.append(float(fid["value"]))
    summary = {"num_ok": len(values), "num_trials": len(trials)}
    if values:
        summary["fid_mean"] = sum(values) / len(values)
        summary["fid_values"] = values
    return summary


def main() -> None:
    args = parse_args()
    ref_dir = resolve_path(args.fid_style_ref_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs_by_style = load_refs_by_style(ref_dir, args.recursive)
    report: dict[str, Any] = {
        "fid_style_ref_dir": str(ref_dir),
        "seed": args.seed,
        "num_splits": args.num_splits,
        "include_self_check": args.include_self_check,
        "fid_min_images": args.fid_min_images,
        "fid_batch_size": args.fid_batch_size,
        "metrics": {"by_style": {}},
    }

    for style, refs in sorted(refs_by_style.items()):
        print(f"Style {style}: splitting {len(refs)} reference images", flush=True)
        trials = []
        for split_idx in range(args.num_splits):
            group_a, group_b = split_refs(style, refs, args.seed, split_idx)
            staged = stage_split_dirs(style, split_idx, group_a, group_b, output_dir, args.copy_inputs)
            fid = compute_fid(
                staged["group_b"],
                staged["methods"],
                args.device,
                args.fid_weights,
                args.fid_min_images,
                args.fid_batch_size,
            )
            trial = {
                "split": split_idx,
                "num_group_a": len(group_a),
                "num_group_b": len(group_b),
                "group_a": [str(path) for path in group_a],
                "group_b": [str(path) for path in group_b],
                "fid": fid,
            }
            trials.append(trial)
            with (staged["split_dir"] / "metrics_report.json").open("w", encoding="utf-8") as f:
                json.dump(trial, f, indent=2, ensure_ascii=False)

        report["metrics"]["by_style"][style] = {
            "num_refs": len(refs),
            "trials": trials,
            "summary": summarize_style_trials(trials, style),
        }
        if args.include_self_check:
            report["metrics"]["by_style"][style]["self_check"] = compute_reference_self_check(style, refs, args, output_dir)

    report_path = output_dir / "fid_ref_baseline_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved FID reference baseline report to {report_path}")


if __name__ == "__main__":
    main()
