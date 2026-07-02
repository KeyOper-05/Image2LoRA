#!/usr/bin/env python3
"""Run Image2LoRA inference for all prompt/style-reference pairs.

Default task:
    prompts.md (8 prompts) x style_selected_8 (8 references) -> 64 images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class StyleImage:
    path: Path
    style_id: str


@dataclass(frozen=True)
class BatchRecord:
    prompt_index: int
    prompt: str
    style_index: int
    style_id: str
    style_image: str
    seed: int
    output_image: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all prompt x style reference pairs with Image2LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("inputs")
    input_group.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "prompts.md")
    input_group.add_argument("--style_dir", type=Path, default=PROJECT_ROOT / "style_selected_8")

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "prompt_style_selected_8")
    output_group.add_argument("--manifest", type=Path, default=None)
    output_group.add_argument("--skip_existing", action="store_true")
    output_group.add_argument("--dry_run", action="store_true")

    infer_group = parser.add_argument_group("infer options")
    infer_group.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    infer_group.add_argument("--checkpoint_dir", type=str, required=True)
    infer_group.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    infer_group.add_argument("--num_inference_steps", type=int, default=30)
    infer_group.add_argument("--guidance_scale", type=float, default=7.5)
    infer_group.add_argument("--seed", type=int, default=42)
    infer_group.add_argument(
        "--seed_mode",
        choices=("prompt", "case", "fixed"),
        default="prompt",
        help="prompt: same prompt uses the same seed across styles; case: every pair uses a unique seed; fixed: all pairs use --seed.",
    )
    infer_group.add_argument("--down_dim", type=int, default=64)
    infer_group.add_argument("--up_dim", type=int, default=32)
    infer_group.add_argument("--rank", type=int, default=1)
    infer_group.add_argument("--decoder_blocks", type=int, default=8)
    infer_group.add_argument("--sample_iters", type=int, default=4)
    infer_group.add_argument("--resolution", type=int, default=512)
    infer_group.add_argument("--lora_scale", type=float, default=1.0)
    infer_group.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        if prompt.startswith(("- ", "* ")):
            prompt = prompt[2:].strip()
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def discover_styles(style_dir: Path) -> list[StyleImage]:
    styles: list[StyleImage] = []
    for path in sorted(style_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        style_id = path.name.split("____", 1)[0] if "____" in path.name else path.stem
        styles.append(StyleImage(path=path, style_id=style_id))
    if not styles:
        raise ValueError(f"No style reference images found in {style_dir}")
    return styles


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:max_len] or "prompt"


def output_path(output_dir: Path, prompt_index: int, prompt: str, style: StyleImage) -> Path:
    prompt_slug = slugify(prompt)
    return output_dir / f"p{prompt_index:02d}_{prompt_slug}__{style.style_id}.png"


def seed_for_case(base_seed: int, seed_mode: str, prompt_index: int, style_index: int, num_prompts: int) -> int:
    if seed_mode == "fixed":
        return base_seed
    if seed_mode == "case":
        return base_seed + (style_index - 1) * num_prompts + (prompt_index - 1)
    return base_seed + (prompt_index - 1)


def update_lora_weights(network, weight_list) -> None:
    for weight, lora_layer in zip(weight_list, network.unet_loras):
        if weight.dim() == 3:
            weight = weight.view(weight.size(0), -1)
        elif weight.dim() == 2 and weight.size(0) == 1:
            weight = weight.view(-1)
        lora_layer.update_weight(weight)


def write_manifest(path: Path, records: list[BatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_models(args: argparse.Namespace, device, dtype):
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer

    from image2lora.models.encoder import DINOv2Encoder
    from image2lora.models.hypernet import ImageHyperDream
    from image2lora.models.lora import create_network

    model_path = resolve_path(args.pretrained_model)
    load_kw = {"local_files_only": True} if model_path.is_dir() else {}

    tokenizer = CLIPTokenizer.from_pretrained(str(model_path), subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(str(model_path), subfolder="text_encoder", torch_dtype=dtype, **load_kw)
    vae = AutoencoderKL.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype, **load_kw)
    unet = UNet2DConditionModel.from_pretrained(str(model_path), subfolder="unet", torch_dtype=dtype, **load_kw)
    scheduler = DDIMScheduler.from_pretrained(str(model_path), subfolder="scheduler", **load_kw)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    network = create_network(
        args.lora_scale,
        args.rank,
        1.0,
        text_encoder,
        unet,
        down_dim=args.down_dim,
        up_dim=args.up_dim,
        is_train=True,
        train_unet=True,
        train_text_encoder=False,
    )

    checkpoint_dir = resolve_path(args.checkpoint_dir)
    lora_path = checkpoint_dir / "lora_aux.safetensors"
    if lora_path.exists():
        network.load_state_dict(load_file(str(lora_path)), strict=False)
        print(f"Loaded LightLoRA aux from {lora_path}", flush=True)
    network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)

    lora_weight_dim = (args.down_dim + args.up_dim) * args.rank
    hypernetwork = ImageHyperDream(
        image_feat_dim=768,
        weight_dim=lora_weight_dim,
        weight_num=len(network.unet_loras),
        decoder_blocks=args.decoder_blocks,
        sample_iters=args.sample_iters,
    )
    hyper_path = checkpoint_dir / "hypernetwork.safetensors"
    hypernetwork.load_state_dict(load_file(str(hyper_path)))
    hypernetwork.eval()
    print(f"Loaded hypernetwork from {hyper_path}", flush=True)

    dinov2_path = resolve_path(args.dinov2_model_path)
    image_encoder = DINOv2Encoder(model_path=str(dinov2_path), local_files_only=dinov2_path.is_dir())
    image_encoder.eval()

    network.to(device, dtype=dtype)
    hypernetwork.to(device, dtype=dtype)
    image_encoder.to(device)
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    return network, hypernetwork, image_encoder, pipe


def generate_lora_for_style(args: argparse.Namespace, style: StyleImage, network, hypernetwork, image_encoder, device, dtype) -> None:
    from PIL import Image
    from torchvision import transforms

    ref_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ref_pil = Image.open(style.path).convert("RGB")
    ref_tensor = ref_transform(ref_pil).unsqueeze(0).to(device)

    ref_features = image_encoder.encode(ref_tensor).to(dtype=dtype)
    _weight_embedding, weight_list = hypernetwork(ref_features)
    update_lora_weights(network, weight_list)


def main() -> None:
    args = parse_args()
    prompts_path = resolve_path(args.prompts)
    style_dir = resolve_path(args.style_dir)
    output_dir = resolve_path(args.output_dir)
    manifest = resolve_path(args.manifest) if args.manifest is not None else output_dir / "manifest.json"

    prompts = read_prompts(prompts_path)
    styles = discover_styles(style_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts) * len(styles)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}", flush=True)
    print(f"Loaded {len(styles)} style references from {style_dir}", flush=True)
    print(f"Planned generations: {total}", flush=True)

    records: list[BatchRecord] = []
    if args.dry_run:
        for style_index, style in enumerate(styles, start=1):
            for prompt_index, prompt in enumerate(prompts, start=1):
                seed = seed_for_case(args.seed, args.seed_mode, prompt_index, style_index, len(prompts))
                output = output_path(output_dir, prompt_index, prompt, style)
                records.append(
                    BatchRecord(
                        prompt_index=prompt_index,
                        prompt=prompt,
                        style_index=style_index,
                        style_id=style.style_id,
                        style_image=str(style.path),
                        seed=seed,
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

    case_index = 0
    for style_index, style in enumerate(styles, start=1):
        print(f"Style [{style_index}/{len(styles)}] {style.style_id}: {style.path.name}", flush=True)
        with torch.no_grad():
            generate_lora_for_style(args, style, network, hypernetwork, image_encoder, device, dtype)

        for prompt_index, prompt in enumerate(prompts, start=1):
            case_index += 1
            seed = seed_for_case(args.seed, args.seed_mode, prompt_index, style_index, len(prompts))
            output = output_path(output_dir, prompt_index, prompt, style)
            status = "generated"

            if args.skip_existing and output.exists():
                status = "skipped_existing"
                print(f"[{case_index}/{total}] skip existing {output.name}", flush=True)
            else:
                print(f"[{case_index}/{total}] p{prompt_index:02d} + {style.style_id} -> {output.name}", flush=True)
                generator = torch.Generator(device=device).manual_seed(seed)
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

            records.append(
                BatchRecord(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    style_index=style_index,
                    style_id=style.style_id,
                    style_image=str(style.path),
                    seed=seed,
                    output_image=str(output),
                    status=status,
                )
            )
            write_manifest(manifest, records)

    print(f"Wrote {len(records)} records to {manifest}", flush=True)


if __name__ == "__main__":
    main()
