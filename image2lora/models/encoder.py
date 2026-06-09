"""DINOv2 image encoder for extracting semantic style features."""

import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# torch.hub 名称 -> HuggingFace 仓库
DINOV2_HF_MAP = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder producing patch-level features for the hypernetwork."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        model_path: Optional[str] = None,
        image_size: int = 518,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.model_path = self._resolve_model_path(model_name, model_path)
        self.local_files_only = local_files_only or os.path.isdir(self.model_path)

        self.model, self.feat_dim = self._load_model()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _resolve_model_path(self, model_name: str, model_path: Optional[str]) -> str:
        if model_path:
            return model_path
        return DINOV2_HF_MAP.get(model_name, "facebook/dinov2-base")

    def _load_model(self):
        from transformers import Dinov2Model

        print(
            f"Loading DINOv2 from {self.model_path} "
            f"(local_files_only={self.local_files_only})"
        )
        model = Dinov2Model.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        feat_dim = model.config.hidden_size
        return model, feat_dim

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        # last_hidden_state: [B, 1+N_patches, D]，去掉 CLS token
        return outputs.last_hidden_state[:, 1:, :]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1] or [-1, 1]
        Returns:
            features: (B, L, D) patch tokens
        """
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = torch.nn.functional.interpolate(
                images, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return self._extract_patch_tokens(images)

    def encode_pil(self, pil_image: Image.Image, device: torch.device) -> torch.Tensor:
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            return self._extract_patch_tokens(tensor)
