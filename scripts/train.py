#!/usr/bin/env python3
"""
Image2LoRA 训练脚本 (SD 1.5)
参考 Video2LoRA 的端到端超网络 + LightLoRA 训练流程。
"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from image2lora.data.dataset import ImagePairDataset, collate_fn
from image2lora.models.encoder import DINOv2Encoder
from image2lora.models.hypernet import ImageHyperDream
from image2lora.models.lora import LoRANetwork, create_network

check_min_version("0.27.0")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Image2LoRA training")
    parser.add_argument("--config", type=str, default="configs/train_sd15.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--train_data_meta", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, args.config))
    for key in ["pretrained_model_name_or_path", "train_data_meta", "train_data_dir",
                "output_dir", "train_batch_size", "learning_rate", "max_train_steps",
                "resume_from_checkpoint"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    # 相对路径转为绝对路径，确保从本地目录加载
    model_path = cfg.pretrained_model_name_or_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if os.path.isdir(model_path):
        cfg.pretrained_model_name_or_path = model_path
        cfg.local_files_only = True
    else:
        cfg.local_files_only = False
    return cfg


def save_checkpoint(output_dir, network, hypernetwork, global_step, weight_dtype):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    network.save_weights(os.path.join(ckpt_dir, "lora_aux.safetensors"), weight_dtype)
    from safetensors.torch import save_file
    hyper_sd = {k: v.detach().cpu().to(weight_dtype) for k, v in hypernetwork.state_dict().items()}
    save_file(hyper_sd, os.path.join(ckpt_dir, "hypernetwork.safetensors"))
    logger.info(f"Saved checkpoint to {ckpt_dir}")


def resolve_resume_checkpoint(resume_path: str, output_dir: str) -> str:
    if resume_path in (None, "", "latest"):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if not ckpts:
            raise ValueError(f"No checkpoint found in {output_dir}")
        return os.path.join(output_dir, ckpts[-1])
    if os.path.isdir(resume_path):
        return resume_path
    if os.path.isdir(os.path.join(output_dir, resume_path)):
        return os.path.join(output_dir, resume_path)
    raise ValueError(f"Checkpoint not found: {resume_path}")


def load_checkpoint(ckpt_dir, network, hypernetwork):
    from safetensors.torch import load_file
    lora_path = os.path.join(ckpt_dir, "lora_aux.safetensors")
    hyper_path = os.path.join(ckpt_dir, "hypernetwork.safetensors")
    network.load_state_dict(load_file(lora_path), strict=False)
    hypernetwork.load_state_dict(load_file(hyper_path))
    global_step = int(os.path.basename(ckpt_dir).split("-")[-1])
    logger.info(f"Resumed weights from {ckpt_dir} (step={global_step})")
    return global_step


def update_lora_weights(network, weight_list):
    actual = network.module if hasattr(network, "module") else network
    for weight, lora_layer in zip(weight_list, actual.unet_loras):
        if weight.dim() == 3:
            weight = weight.view(weight.size(0), -1)
        elif weight.dim() == 2 and weight.size(0) == 1:
            weight = weight.view(-1)
        lora_layer.update_weight(weight)


def main():
    args = parse_args()
    cfg = load_config(args)

    logging_dir = os.path.join(cfg.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir),
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

    # ---- 加载 SD 1.5 组件 ----
    model_path = cfg.pretrained_model_name_or_path
    load_kwargs = {"local_files_only": cfg.get("local_files_only", False)}
    if accelerator.is_main_process:
        logger.info(f"Loading SD 1.5 from: {model_path} (local_files_only={load_kwargs['local_files_only']})")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kwargs)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **load_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kwargs)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # ---- LightLoRA ----
    network = create_network(
        1.0, cfg.rank, cfg.network_alpha,
        text_encoder, unet,
        down_dim=cfg.down_dim, up_dim=cfg.up_dim,
        is_train=True, train_unet=True, train_text_encoder=False,
    )
    network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)

    lora_weight_dim = (cfg.down_dim + cfg.up_dim) * cfg.rank
    actual_network = network
    hypernetwork = ImageHyperDream(
        image_feat_dim=768,
        weight_dim=lora_weight_dim,
        weight_num=len(actual_network.unet_loras),
        decoder_blocks=cfg.decoder_blocks,
        sample_iters=cfg.sample_iters,
    )
    hypernetwork.set_lilora(actual_network.unet_loras)
    hypernetwork.set_device(accelerator.device)

    # ---- DINOv2 编码器 (冻结) ----
    dinov2_path = cfg.get("dinov2_model_path")
    if dinov2_path and not os.path.isabs(dinov2_path):
        dinov2_path = os.path.join(PROJECT_ROOT, dinov2_path)
    image_encoder = DINOv2Encoder(
        model_name=cfg.dinov2_model,
        model_path=dinov2_path,
        local_files_only=bool(dinov2_path and os.path.isdir(dinov2_path)),
    )
    image_encoder.eval()

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        hypernetwork.enable_gradient_checkpointing()

    # ---- 优化器 ----
    trainable_params = network.prepare_optimizer_params(
        text_encoder_lr=None, unet_lr=cfg.learning_rate / 2, default_lr=cfg.learning_rate
    )
    hyper_params = [{"params": list(hypernetwork.parameters()), "lr": cfg.learning_rate}]
    optimizer = torch.optim.AdamW(
        trainable_params + hyper_params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # ---- 数据集 ----
    meta_path = cfg.train_data_meta
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(PROJECT_ROOT, meta_path)
    data_root = cfg.train_data_dir
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root)

    train_dataset = ImagePairDataset(
        meta_path=meta_path,
        data_root=data_root,
        resolution=cfg.resolution,
        text_drop_ratio=cfg.text_drop_ratio,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps > 0:
        max_train_steps = cfg.max_train_steps
    else:
        max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    network, hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, hypernetwork, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device)

    global_step = 0
    if cfg.get("resume_from_checkpoint"):
        ckpt_dir = resolve_resume_checkpoint(cfg.resume_from_checkpoint, cfg.output_dir)
        global_step = load_checkpoint(
            ckpt_dir,
            accelerator.unwrap_model(network),
            accelerator.unwrap_model(hypernetwork),
        )
        # 恢复学习率调度器到对应 step（优化器动量未保存，会有轻微跳变）
        for _ in range(global_step):
            lr_scheduler.step()

    if accelerator.is_main_process:
        accelerator.init_trackers("image2lora", config=OmegaConf.to_container(cfg))

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** Running Image2LoRA training *****")
    logger.info(f"  Num pairs = {len(train_dataset)}")
    logger.info(f"  Num LightLoRA layers = {len(actual_network.unet_loras)}")
    logger.info(f"  Max steps = {max_train_steps}")

    for epoch in range(cfg.num_train_epochs):
        network.train()
        hypernetwork.train()
        train_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(network, hypernetwork):
                ref_images = batch["ref_image"].to(accelerator.device)
                tgt_images = batch["tgt_image"].to(accelerator.device, dtype=weight_dtype)

                # ---- HyperNetwork: ref -> LightLoRA weights ----
                with torch.no_grad():
                    ref_features = image_encoder.encode(ref_images)
                # 超网络保持 fp32 权重，避免 fp16 梯度与 GradScaler 冲突
                _, weight_list = hypernetwork(ref_features.float())
                update_lora_weights(network, weight_list)

                # ---- VAE encode target ----
                with torch.no_grad():
                    latents = vae.encode(tgt_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # ---- Text encode ----
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch["caption"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoder_hidden_states = text_encoder(
                        text_inputs.input_ids.to(accelerator.device)
                    )[0]

                # ---- Diffusion noise prediction loss ----
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(network.parameters()) + list(hypernetwork.parameters()),
                        cfg.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()

                if global_step % cfg.logging_steps == 0:
                    avg_loss = train_loss / cfg.logging_steps
                    train_loss = 0.0
                    logs = {"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            cfg.output_dir,
                            accelerator.unwrap_model(network),
                            accelerator.unwrap_model(hypernetwork),
                            global_step, weight_dtype,
                        )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            cfg.output_dir,
            accelerator.unwrap_model(network),
            accelerator.unwrap_model(hypernetwork),
            global_step, weight_dtype,
        )
        logger.info("Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
