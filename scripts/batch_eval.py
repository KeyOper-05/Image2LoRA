#!/usr/bin/env python3
"""Batch evaluation for Image2LoRA style groups.

This script follows scripts/batch_infer.py naming rules:
    content.ext&&style.ext.jpg

For each style reference, it evaluates the generated images that already exist.
Styles with zero generated images are ignored by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (  # noqa: E402
    EvalCase,
    compute_fid,
    compute_style_loss,
    stage_metric_dirs,
    summarize_style_loss,
    validate_cases,
)


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
class BatchEvalCase:
    prompt: str
    style_ref: Path
    generated: Path
    style: str
    caption: str
    content_image: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate batch inference outputs per style with Style Loss and FID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_argument_group("inputs")
    input_group.add_argument(
        "--batch_manifest",
        type=Path,
        default=None,
        help="JSON manifest written by scripts/batch_infer.py.",
    )
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
    input_group.add_argument(
        "--generated_dir",
        type=Path,
        default=None,
        help="Folder containing generated images named content.ext&&style.ext.jpg.",
    )
    input_group.add_argument("--recursive", action="store_true", help="Search content/style folders recursively.")
    input_group.add_argument(
        "--prompt_template",
        type=str,
        default="{caption}",
        help="Used only without --batch_manifest. Fields: {type}, {caption}, {style}, {content_filename}, {style_filename}.",
    )
    input_group.add_argument(
        "--allow_missing",
        action="store_true",
        help="Deprecated no-op. Missing generated images are allowed by default.",
    )
    input_group.add_argument(
        "--require_complete",
        action="store_true",
        help="Require every style to have generated images for every caption.",
    )

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--output_dir", type=Path, required=True, help="Evaluation output directory.")
    output_group.add_argument(
        "--eval_manifest",
        type=Path,
        default=None,
        help="JSONL manifest written for metric inputs. Defaults to <output_dir>/eval_manifest.jsonl.",
    )

    metric_group = parser.add_argument_group("metrics")
    metric_group.add_argument("--device", default="cuda", help="Device string for metrics.")
    metric_group.add_argument("--skip_fid", action="store_true", help="Skip pytorch-fid.")
    metric_group.add_argument(
        "--fid_weights",
        type=Path,
        default=None,
        help="Optional local pytorch-fid Inception weights. Avoids network download.",
    )
    metric_group.add_argument("--style_loss_image_size", type=int, default=256)
    metric_group.add_argument(
        "--style_loss_vgg_weights",
        type=Path,
        default=None,
        help="Optional local torchvision VGG19 state_dict for Style Loss.",
    )
    metric_group.add_argument(
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


def make_prompt(template: str, content: ContentImage, style: StyleImage) -> str:
    return template.format(
        type=content.content_type,
        caption=content.caption,
        style=style.style,
        content_filename=content.path.name,
        style_filename=style.path.name,
    )


def load_cases_from_batch_manifest(path: Path) -> list[BatchEvalCase]:
    manifest = resolve_path(path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{manifest} must be the JSON array written by scripts/batch_infer.py")

    cases = []
    for idx, record in enumerate(data, start=1):
        status = str(record.get("status", ""))
        if status.startswith("failed") or status == "dry_run":
            continue
        try:
            generated = resolve_path(Path(record["output_image"]))
            style_ref = resolve_path(Path(record["style_image"]))
        except KeyError as exc:
            raise ValueError(f"Missing required field {exc} in {manifest} record {idx}") from exc
        content_image = record.get("content_image")
        cases.append(
            BatchEvalCase(
                prompt=str(record.get("prompt") or record.get("caption") or generated.stem),
                style_ref=style_ref,
                generated=generated,
                style=str(record.get("style") or parse_style(style_ref).style),
                caption=str(record.get("caption") or ""),
                content_image=resolve_path(Path(content_image)) if content_image else None,
            )
        )
    return cases


def load_cases_from_dirs(args: argparse.Namespace) -> list[BatchEvalCase]:
    if args.generated_dir is None:
        raise ValueError("Pass --generated_dir when not using --batch_manifest.")
    if args.input_dir is None and (args.content_dir is None or args.style_dir is None):
        raise ValueError("Pass --input_dir, or pass both --content_dir and --style_dir.")

    input_dir = resolve_path(args.input_dir) if args.input_dir is not None else None
    content_dir = resolve_path(args.content_dir) if args.content_dir is not None else input_dir
    style_dir = resolve_path(args.style_dir) if args.style_dir is not None else input_dir
    if content_dir is None or style_dir is None:
        raise ValueError("Could not resolve content/style folders.")

    generated_dir = resolve_path(args.generated_dir)
    contents = [item for path in iter_image_files(content_dir, args.recursive) if (item := parse_content(path)) is not None]
    styles = [item for path in iter_image_files(style_dir, args.recursive) if (item := parse_style(path)) is not None]
    if not contents:
        raise ValueError(f"No content images found in {content_dir}. Expected filenames like type_caption.ext")
    if not styles:
        raise ValueError(f"No style images found in {style_dir}. Expected filenames like style____*.jpg")

    cases = []
    for style in styles:
        for content in contents:
            generated = generated_dir / f"{content.path.name}&&{style.path.name}.jpg"
            cases.append(
                BatchEvalCase(
                    prompt=make_prompt(args.prompt_template, content, style),
                    style_ref=style.path,
                    generated=generated,
                    style=style.style,
                    caption=content.caption,
                    content_image=content.path,
                )
            )
    return cases


def check_caption_coverage(cases: list[BatchEvalCase], require_complete: bool) -> list[BatchEvalCase]:
    by_style: dict[str, list[BatchEvalCase]] = defaultdict(list)
    for case in cases:
        by_style[case.style].append(case)

    valid_cases = []
    errors = []
    for style, style_cases in sorted(by_style.items()):
        existing = [case for case in style_cases if case.generated.exists()]
        missing = [case for case in style_cases if not case.generated.exists()]
        if missing and require_complete:
            missing_outputs = [case.generated.name for case in missing[:20]]
            suffix = f" ... and {len(missing) - 20} more" if len(missing) > 20 else ""
            errors.append(f"{style} missing {len(missing)} generated images: {missing_outputs}{suffix}")

        expected_refs = len({case.style_ref for case in style_cases})
        expected_captions = len({case.caption for case in style_cases})
        existing_refs = len({case.style_ref for case in existing})
        existing_captions = len({case.caption for case in existing})
        print(
            f"Style {style}: {len(existing)}/{len(style_cases)} generated images found "
            f"({existing_refs}/{expected_refs} refs, {existing_captions}/{expected_captions} captions)",
            flush=True,
        )
        if existing:
            valid_cases.extend(existing)
        else:
            print(f"Style {style}: skipped because no generated images were found", flush=True)

    if errors:
        message = "\n".join(errors)
        raise ValueError(f"Missing generated images. Re-run batch inference or remove --require_complete.\n{message}")
    return valid_cases


def to_eval_case(case: BatchEvalCase) -> EvalCase:
    return EvalCase(
        prompt=case.prompt,
        style_ref=case.style_ref,
        generated=case.generated,
        method=case.style,
        content=None,
    )


def write_eval_manifest(path: Path, cases: list[BatchEvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            record = {
                "prompt": case.prompt,
                "style_ref": str(case.style_ref),
                "generated": str(case.generated),
                "method": case.style,
            }
            if case.content_image is not None:
                record["content"] = str(case.content_image)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_style_loss_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["style", "prompt", "generated", "style_ref", "style_loss_gram", "style_loss_adain"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_style_loss_result(style_loss_result: dict[str, Any], style: str) -> dict[str, Any]:
    if style_loss_result.get("status") != "ok":
        return {**style_loss_result, "summary": {}}

    rows = [row for row in style_loss_result.get("rows") or [] if row.get("method") == style]
    filtered = {**style_loss_result, "rows": rows}
    filtered["summary"] = summarize_style_loss(filtered)
    return filtered


def evaluate_style_group(
    style: str,
    cases: list[BatchEvalCase],
    style_loss_result: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    eval_cases = validate_cases([to_eval_case(case) for case in cases])
    style_output_dir = output_dir / "by_style" / style
    style_output_dir.mkdir(parents=True, exist_ok=True)

    staged = stage_metric_dirs(eval_cases, style_output_dir, args.copy_inputs)
    if args.skip_fid:
        fid_result = {style: {"status": "skipped", "reason": "--skip_fid"}}
    else:
        fid_result = compute_fid(staged["style"], staged["methods"], args.device, args.fid_weights)
    style_style_loss_result = filter_style_loss_result(style_loss_result, style)

    style_report = {
        "style": style,
        "style_ref": str(cases[0].style_ref),
        "num_cases": len(eval_cases),
        "metrics": {
            "fid": fid_result,
            "style_loss": style_style_loss_result,
        },
    }
    with (style_output_dir / "metrics_report.json").open("w", encoding="utf-8") as f:
        json.dump(style_report, f, indent=2, ensure_ascii=False)
    return style_report


def aggregate_style_loss_rows(style_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in style_reports:
        style = str(report["style"])
        for row in report["metrics"]["style_loss"].get("rows") or []:
            rows.append(
                {
                    "style": style,
                    "prompt": row["prompt"],
                    "generated": row["generated"],
                    "style_ref": row["style_ref"],
                    "style_loss_gram": row["style_loss_gram"],
                    "style_loss_adain": row["style_loss_adain"],
                }
            )
    return rows


def summarize_batch(style_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_style = {}
    for report in style_reports:
        style = str(report["style"])
        fid_for_style = report["metrics"]["fid"].get(style, {})
        style_loss_metric = report["metrics"]["style_loss"]
        style_loss_summary = style_loss_metric.get("summary", {}).get(style, {})
        by_style[style] = {
            "num_cases": report["num_cases"],
            "style_ref": report["style_ref"],
            "fid": fid_for_style,
            "style_loss": {
                "status": style_loss_metric.get("status"),
                "reason": style_loss_metric.get("reason"),
                "install": style_loss_metric.get("install"),
                "summary": style_loss_summary,
            },
        }
    return by_style


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_manifest = resolve_path(args.eval_manifest) if args.eval_manifest is not None else output_dir / "eval_manifest.jsonl"

    if args.batch_manifest is not None:
        cases = load_cases_from_batch_manifest(args.batch_manifest)
    else:
        cases = load_cases_from_dirs(args)
    if not cases:
        raise ValueError("No generated cases found to evaluate.")

    cases = check_caption_coverage(cases, args.require_complete)
    if not cases:
        raise ValueError("No generated images found to evaluate.")
    write_eval_manifest(eval_manifest, cases)

    by_style: dict[str, list[BatchEvalCase]] = defaultdict(list)
    for case in cases:
        by_style[case.style].append(case)

    all_eval_cases = validate_cases([to_eval_case(case) for case in cases if case.generated.exists()])
    style_args = argparse.Namespace(
        device=args.device,
        style_loss_image_size=args.style_loss_image_size,
        style_loss_vgg_weights=args.style_loss_vgg_weights,
    )
    print(f"Computing Style Loss for {len(all_eval_cases)} generated images", flush=True)
    style_loss_result = compute_style_loss(all_eval_cases, style_args)

    style_reports = []
    for style, style_cases in sorted(by_style.items()):
        print(f"Evaluating style {style}: {len(style_cases)} caption images", flush=True)
        style_reports.append(evaluate_style_group(style, style_cases, style_loss_result, args, output_dir))

    style_loss_rows = aggregate_style_loss_rows(style_reports)
    write_style_loss_csv(style_loss_rows, output_dir / "style_loss.csv")

    report = {
        "num_styles": len(style_reports),
        "num_cases": sum(report["num_cases"] for report in style_reports),
        "eval_manifest": str(eval_manifest),
        "metrics": {
            "by_style": summarize_batch(style_reports),
            "lower_is_better": ["fid", "style_loss_gram", "style_loss_adain"],
        },
    }
    report_path = output_dir / "batch_metrics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved batch metric report to {report_path}")


if __name__ == "__main__":
    main()
