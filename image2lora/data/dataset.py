"""Paired image dataset for Image2LoRA training."""

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagePairDataset(Dataset):
    """
    成对图像数据集，每条样本包含:
      - ref_image: 风格参考图 I_ref
      - tgt_image: 风格化目标图 I_tgt
      - caption: 文本描述

    metadata.jsonl 每行格式:
    {"ref_image": "styles/xxx.jpg", "tgt_image": "targets/yyy.jpg", "caption": "...", "class": "watercolor"}
    """

    def __init__(
        self,
        meta_path: str,
        data_root: str,
        resolution: int = 512,
        text_drop_ratio: float = 0.1,
        center_crop: bool = True,
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.text_drop_ratio = text_drop_ratio

        print(f"Loading metadata from {meta_path}")
        self.samples: List[Dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} image pairs")

        # 按 class 分组，用于同类别内随机采样 ref（Video2LoRA 持续学习策略）
        self.class_to_indices: Dict[str, List[int]] = {}
        for i, item in enumerate(self.samples):
            cls = item.get("class", "default")
            self.class_to_indices.setdefault(cls, []).append(i)

        if center_crop:
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path: str) -> Image.Image:
        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def sample_ref_index(self, idx: int) -> int:
        """同类别内随机选另一张作为 ref（持续学习）。"""
        cls = self.samples[idx].get("class", "default")
        candidates = [i for i in self.class_to_indices[cls] if i != idx]
        if not candidates:
            return idx
        return random.choice(candidates)

    def __getitem__(self, idx: int):
        max_retries = 10
        for _ in range(max_retries):
            try:
                item = self.samples[idx]
                tgt_image = self._load_image(item["tgt_image"])
                caption = item.get("caption", "")

                # 优先使用显式 ref_image；否则同 class 随机采样
                if "ref_image" in item and item["ref_image"]:
                    ref_image = self._load_image(item["ref_image"])
                else:
                    ref_idx = self.sample_ref_index(idx)
                    ref_image = self._load_image(self.samples[ref_idx]["tgt_image"])

                if random.random() < self.text_drop_ratio:
                    caption = ""

                return {
                    "ref_image": self.transform(ref_image),
                    "tgt_image": self.transform(tgt_image),
                    "caption": caption,
                    "idx": idx,
                }
            except Exception as e:
                print(f"Dataset retry ({e}), idx={idx}")
                idx = random.randint(0, len(self.samples) - 1)
        raise RuntimeError("Failed to load sample after retries")


def collate_fn(batch):
    return {
        "ref_image": torch.stack([b["ref_image"] for b in batch]),
        "tgt_image": torch.stack([b["tgt_image"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
