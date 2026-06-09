#!/usr/bin/env python3
"""i2L 架构训练: DINOv2 (冻结) + 逐层双层 MLP -> 标准 LoRA -> SD 1.5 扩散损失。"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

I2L_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(I2L_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, I2L_ROOT)

from i2l_sd15.models.generator import Image2LoRAGenerator
from i2l_sd15.models.lora import create_network
from image2lora.data.dataset import ImagePairDataset, collate_fn
from image2lora.models.encoder import DINOv2Encoder

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=os.path.join(I2L_ROOT, "configs/train_sd15.yaml"))
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--train_data_meta", type=str, default=None)
    p.add_argument("--train_data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def load_config(args):
    cfg = OmegaConf.load(args.config)
    for key in ["pretrained_model_name_or_path", "train_data_meta", "train_data_dir",
                "output_dir", "train_batch_size", "learning_rate", "max_train_steps",
                "resume_from_checkpoint"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    model_path = cfg.pretrained_model_name_or_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    cfg.pretrained_model_name_or_path = model_path
    cfg.local_files_only = os.path.isdir(model_path)
    return cfg


def resolve_resume_checkpoint(resume_path, output_dir):
    if resume_path in (None, "", "latest"):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if not ckpts:
            raise ValueError(f"No checkpoint in {output_dir}")
        return os.path.join(output_dir, ckpts[-1])
    if os.path.isdir(resume_path):
        return resume_path
    alt = os.path.join(output_dir, resume_path)
    if os.path.isdir(alt):
        return alt
    raise ValueError(f"Checkpoint not found: {resume_path}")


def save_checkpoint(output_dir, generator, global_step, weight_dtype):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    sd = {k: v.detach().cpu().to(weight_dtype) for k, v in generator.state_dict().items()}
    save_file(sd, os.path.join(ckpt_dir, "i2l_generator.safetensors"))
    logger.info(f"Saved checkpoint to {ckpt_dir}")


def main():
    args = parse_args()
    cfg = load_config(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=os.path.join(cfg.output_dir, "logs")),
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = cfg.pretrained_model_name_or_path
    load_kw = {"local_files_only": cfg.local_files_only}
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kw)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **load_kw)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **load_kw)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kw)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    network = create_network(unet, rank=cfg.rank, alpha=cfg.network_alpha, multiplier=cfg.lora_scale)
    network.apply_to(unet)

    generator = Image2LoRAGenerator(
        layer_specs=network.layer_specs,
        feat_dim=768,
        hidden_dim=cfg.hidden_dim,
    )

    dinov2_path = cfg.get("dinov2_model_path")
    if dinov2_path and not os.path.isabs(dinov2_path):
        dinov2_path = os.path.join(PROJECT_ROOT, dinov2_path)
    image_encoder = DINOv2Encoder(
        model_name=cfg.dinov2_model,
        model_path=dinov2_path,
        local_files_only=bool(dinov2_path and os.path.isdir(dinov2_path)),
    )
    image_encoder.eval()

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    meta_path = cfg.train_data_meta
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(PROJECT_ROOT, meta_path)
    data_root = cfg.train_data_dir
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root)

    train_dataset = ImagePairDataset(
        meta_path=meta_path, data_root=data_root,
        resolution=cfg.resolution, text_drop_ratio=cfg.text_drop_ratio,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=max_train_steps,
    )

    generator, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        generator, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device)

    global_step = 0
    if cfg.get("resume_from_checkpoint"):
        ckpt_dir = resolve_resume_checkpoint(cfg.resume_from_checkpoint, cfg.output_dir)
        gen_sd = load_file(os.path.join(ckpt_dir, "i2l_generator.safetensors"))
        accelerator.unwrap_model(generator).load_state_dict(gen_sd)
        global_step = int(os.path.basename(ckpt_dir).split("-")[-1])
        for _ in range(global_step):
            lr_scheduler.step()
        logger.info(f"Resumed from {ckpt_dir} (step={global_step})")

    if accelerator.is_main_process:
        accelerator.init_trackers("i2l_sd15", config=OmegaConf.to_container(cfg))

    progress_bar = tqdm(
        range(global_step, max_train_steps), initial=global_step, total=max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** i2L SD1.5 training *****")
    logger.info(f"  Num pairs = {len(train_dataset)}")
    logger.info(f"  LoRA layers = {len(network.unet_loras)}")
    logger.info(f"  Max steps = {max_train_steps}")

    for epoch in range(cfg.num_train_epochs):
        generator.train()
        train_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(generator):
                ref_images = batch["ref_image"].to(accelerator.device)
                tgt_images = batch["tgt_image"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    ref_features = image_encoder.encode(ref_images)
                lora_weights = generator(ref_features.float())
                network.set_lora_weights(lora_weights)

                with torch.no_grad():
                    latents = vae.encode(tgt_images).latent_dist.sample() * vae.config.scaling_factor
                    text_inputs = tokenizer(
                        batch["caption"], padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
                    )
                    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(generator.parameters(), cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()

                if global_step % cfg.logging_steps == 0:
                    logs = {"loss": train_loss / cfg.logging_steps, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    train_loss = 0.0
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_checkpoint(cfg.output_dir, accelerator.unwrap_model(generator), global_step, weight_dtype)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(cfg.output_dir, accelerator.unwrap_model(generator), global_step, weight_dtype)
        logger.info("Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
