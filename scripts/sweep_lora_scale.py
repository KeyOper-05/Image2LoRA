#!/usr/bin/env python3
"""Sweep LightLoRA injection strength for one prompt/reference pair.

The LoRA contribution is applied in LoRAModule.forward as:
    output = original_output + delta_output * multiplier

This script keeps the prompt, reference image, seed, and generated LightLoRA
weights fixed, then changes only the LoRA multiplier.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from batch_infer_prompts import (
    StyleImage,
    discover_styles,
    generate_lora_for_style,
    load_models,
    read_prompts,
    resolve_path,
    slugify,
)


@dataclass(frozen=True)
class ScaleRecord:
    prompt: str
    style_id: str
    style_image: str
    seed: int
    lora_multiplier: float
    output_image: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with different LightLoRA injection multipliers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("inputs")
    input_group.add_argument("--prompt", type=str, default=None, help="Prompt to use. Defaults to the first prompt in --prompts.")
    input_group.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "prompts.md")
    input_group.add_argument("--ref_image", type=Path, default=None, help="Reference style image. Defaults to the first image in --style_dir.")
    input_group.add_argument("--style_dir", type=Path, default=PROJECT_ROOT / "style_selected_8")

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "lora_scale_sweep")
    output_group.add_argument("--manifest", type=Path, default=None)
    output_group.add_argument("--grid", type=Path, default=None, help="Grid image path. Defaults to <output_dir>/grid.png.")
    output_group.add_argument("--skip_existing", action="store_true")
    output_group.add_argument("--dry_run", action="store_true")

    sweep_group = parser.add_argument_group("sweep")
    sweep_group.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        help="LoRA multipliers to test. 0.0 is the text-only base model with LoRA disabled.",
    )

    infer_group = parser.add_argument_group("infer options")
    infer_group.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    infer_group.add_argument("--checkpoint_dir", type=str, required=True)
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
    infer_group.add_argument("--lora_scale", type=float, default=1.0, help="Initial multiplier used when constructing LoRA modules.")
    infer_group.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return parser.parse_args()


def choose_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    prompts = read_prompts(resolve_path(args.prompts))
    return prompts[0]


def choose_style(args: argparse.Namespace) -> StyleImage:
    if args.ref_image is not None:
        path = resolve_path(args.ref_image)
        style_id = path.name.split("____", 1)[0] if "____" in path.name else path.stem
        return StyleImage(path=path, style_id=style_id)
    styles = discover_styles(resolve_path(args.style_dir))
    return styles[0]


def set_lora_multiplier(network, scale: float) -> None:
    network.multiplier = scale
    for lora_layer in network.unet_loras:
        lora_layer.multiplier = scale


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace("-", "m").replace(".", "p")


def image_path(output_dir: Path, prompt: str, style: StyleImage, scale: float) -> Path:
    return output_dir / f"{style.style_id}__{slugify(prompt, max_len=70)}__lora_{scale_tag(scale)}.png"


def write_manifest(path: Path, records: list[ScaleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_grid(image_paths: list[Path], scales: list[float], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in image_paths if path.exists()]
    if not images:
        return

    tile_w, tile_h = images[0].size
    label_h = 34
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for idx, image in enumerate(images):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        grid.paste(image, (x, y + label_h))
        draw.text((x + 10, y + 10), f"LoRA multiplier = {scales[idx]:g}", fill=(0, 0, 0), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def main() -> None:
    args = parse_args()
    prompt = choose_prompt(args)
    style = choose_style(args)
    output_dir = resolve_path(args.output_dir)
    manifest = resolve_path(args.manifest) if args.manifest is not None else output_dir / "manifest.json"
    grid_path = resolve_path(args.grid) if args.grid is not None else output_dir / "grid.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ScaleRecord] = []
    planned_outputs = [image_path(output_dir, prompt, style, scale) for scale in args.scales]

    print(f"Prompt: {prompt}", flush=True)
    print(f"Reference style: {style.path}", flush=True)
    print(f"LoRA multipliers: {', '.join(f'{scale:g}' for scale in args.scales)}", flush=True)

    if args.dry_run:
        for scale, output in zip(args.scales, planned_outputs):
            records.append(
                ScaleRecord(
                    prompt=prompt,
                    style_id=style.style_id,
                    style_image=str(style.path),
                    seed=args.seed,
                    lora_multiplier=scale,
                    output_image=str(output),
                    status="dry_run",
                )
            )
        write_manifest(manifest, records)
        print(f"Wrote dry-run manifest with {len(records)} records to {manifest}", flush=True)
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    network, hypernetwork, image_encoder, pipe = load_models(args, device, dtype)

    with torch.no_grad():
        generate_lora_for_style(args, style, network, hypernetwork, image_encoder, device, dtype)

    generated_paths: list[Path] = []
    for idx, (scale, output) in enumerate(zip(args.scales, planned_outputs), start=1):
        status = "generated"
        if args.skip_existing and output.exists():
            status = "skipped_existing"
            print(f"[{idx}/{len(args.scales)}] skip existing {output.name}", flush=True)
        else:
            set_lora_multiplier(network, scale)
            print(f"[{idx}/{len(args.scales)}] LoRA multiplier {scale:g} -> {output.name}", flush=True)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                ).images[0]
            image.save(output)

        generated_paths.append(output)
        records.append(
            ScaleRecord(
                prompt=prompt,
                style_id=style.style_id,
                style_image=str(style.path),
                seed=args.seed,
                lora_multiplier=scale,
                output_image=str(output),
                status=status,
            )
        )
        write_manifest(manifest, records)

    make_grid(generated_paths, args.scales, grid_path)
    print(f"Wrote {len(records)} records to {manifest}", flush=True)
    print(f"Wrote grid to {grid_path}", flush=True)


if __name__ == "__main__":
    main()
