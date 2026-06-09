#!/usr/bin/env python3
"""i2L 架构推理: 参考图 -> DINOv2 -> 逐层 MLP -> LoRA -> SD 1.5 生成。"""

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

I2L_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(I2L_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, I2L_ROOT)

from i2l_sd15.models.generator import Image2LoRAGenerator
from i2l_sd15.models.lora import create_network
from image2lora.models.encoder import DINOv2Encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    p.add_argument("--checkpoint", type=str, required=True, help="i2l_generator.safetensors 或含该文件的 checkpoint 目录")
    p.add_argument("--ref_image", type=str, nargs="+", required=True, help="一张或多张参考图")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, default="output.png")
    p.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--network_alpha", type=float, default=4.0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return p.parse_args()


def resolve_generator_path(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "i2l_generator.safetensors")
    return checkpoint


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_path = args.pretrained_model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    kw = {"local_files_only": True} if os.path.isdir(model_path) else {}

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **kw)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype, **kw)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype, **kw)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype, **kw)
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", **kw)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    network = create_network(unet, rank=args.rank, alpha=args.network_alpha)
    network.apply_to(unet)

    generator = Image2LoRAGenerator(network.layer_specs, feat_dim=768, hidden_dim=args.hidden_dim)
    gen_path = resolve_generator_path(args.checkpoint)
    generator.load_state_dict(load_file(gen_path))
    generator.eval().to(device)
    print(f"Loaded generator from {gen_path}")

    dinov2_path = args.dinov2_model_path
    if not os.path.isabs(dinov2_path):
        dinov2_path = os.path.join(PROJECT_ROOT, dinov2_path)
    image_encoder = DINOv2Encoder(model_path=dinov2_path, local_files_only=os.path.isdir(dinov2_path))
    image_encoder.eval().to(device)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
    ])

    ref_features_list = []
    for ref_path in args.ref_image:
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(PROJECT_ROOT, ref_path)
        tensor = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
        ref_features_list.append(image_encoder.encode(tensor))

    if len(ref_features_list) == 1:
        lora_weights = generator(ref_features_list[0].float())
    else:
        lora_weights = generator.forward_multi_images([f.float() for f in ref_features_list])
        print(f"Merged {len(ref_features_list)} reference images along rank dim")

    network.set_lora_weights(lora_weights)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None,
    ).to(device)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
