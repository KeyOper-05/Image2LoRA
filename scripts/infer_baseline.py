#!/usr/bin/env python3
"""纯 SD 1.5 基线推理，不加载任何 LoRA / Hypernetwork。"""

import argparse
import os
import sys

import torch
from diffusers import StableDiffusionPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, default="output_baseline.png")
    p.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=512)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_path = args.pretrained_model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    kw = {"local_files_only": True} if os.path.isdir(model_path) else {}

    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, safety_checker=None, **kw)
    pipe = pipe.to(device)

    img = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
