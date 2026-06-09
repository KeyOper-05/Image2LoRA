"""Image Hypernetwork: generates LightLoRA weights from DINOv2 image features."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import TransformerBlock
from .lora import LoRAModule


def _get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class WeightDecoder(nn.Module):
    def __init__(self, weight_dim: int, weight_num: int, decoder_blocks: int = 4):
        super().__init__()
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.register_buffer("block_pos_emb", _get_sinusoid_encoding_table(weight_num * 2, weight_dim))

        heads = 1
        while weight_dim % heads == 0 and weight_dim // heads > 64:
            heads *= 2
        heads //= 2

        self.pos_emb_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.decoder_model = nn.ModuleList(
            TransformerBlock(weight_dim, heads, weight_dim // heads, context_dim=weight_dim, gated_ff=False)
            for _ in range(decoder_blocks)
        )
        self.delta_proj = nn.Sequential(nn.LayerNorm(weight_dim), nn.Linear(weight_dim, weight_dim, bias=False))
        self._init_weights()

    def _init_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.delta_proj[1].weight, std=1e-3)

    def forward(self, weight, features):
        pos_emb = self.pos_emb_proj(self.block_pos_emb[:, : weight.size(1)].clone().detach())
        h = weight + pos_emb
        for decoder in self.decoder_model:
            h = decoder(h, context=features)
        return weight + self.delta_proj(h)


class ImageWeightGenerator(nn.Module):
    """Maps DINOv2 patch tokens to LightLoRA weight vectors via iterative refinement."""

    def __init__(
        self,
        image_feat_dim: int,
        weight_dim: int = 96,
        weight_num: int = 64,
        decoder_blocks: int = 8,
        sample_iters: int = 4,
    ):
        super().__init__()
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.sample_iters = sample_iters
        self.feature_proj = nn.Linear(image_feat_dim, weight_dim, bias=False)
        self.decoder_model = WeightDecoder(weight_dim, weight_num, decoder_blocks)

    def encode_features(self, image_features):
        # DINOv2 output: (B, L, C)
        assert image_features.ndim == 3, f"Expected (B, L, C), got {image_features.shape}"
        return image_features

    def decode_weight(self, features, iters=None, weight=None):
        features = self.feature_proj(features)
        if weight is None:
            weight = torch.zeros(
                features.size(0), self.weight_num, self.weight_dim,
                device=features.device, dtype=features.dtype,
            )
        for _ in range(iters or self.sample_iters):
            weight = self.decoder_model(weight, features)
        return weight

    def forward(self, image_features, iters=None, weight=None, ensure_grad=0):
        features = self.encode_features(image_features) + ensure_grad
        return self.decode_weight(features, iters, weight)


class ImageHyperDream(nn.Module):
    """Hypernetwork that generates per-image LightLoRA weights from a reference image."""

    def __init__(
        self,
        image_feat_dim: int = 768,
        weight_dim: int = 96,
        weight_num: int = 64,
        decoder_blocks: int = 8,
        sample_iters: int = 4,
    ):
        super().__init__()
        self.img_weight_generator = ImageWeightGenerator(
            image_feat_dim=image_feat_dim,
            weight_dim=weight_dim,
            weight_num=weight_num,
            decoder_blocks=decoder_blocks,
            sample_iters=sample_iters,
        )
        self.weight_dim = weight_dim
        self.liloras: Dict[str, LoRAModule] = {}
        self.liloras_keys: List[str] = []
        self.gradient_checkpointing = False
        self.device = "cpu"

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def set_lilora(self, liloras):
        self.liloras = liloras
        self.liloras_keys = list(liloras.keys()) if isinstance(liloras, dict) else list(range(len(liloras)))
        print(f"Hypernet: {len(self.liloras_keys)} LightLoRA layers, {self.weight_dim} dims each")

    def set_device(self, device):
        self.device = device

    def gen_weight(self, image_features, iters, weight, ensure_grad=0):
        weights = self.img_weight_generator(image_features, iters, weight, ensure_grad)
        weight_list = [w.squeeze(1) for w in weights.split(1, dim=1)]
        return weights, weight_list

    def forward(self, image_features: torch.Tensor, iters: int = None, weight: torch.Tensor = None):
        if self.training and self.gradient_checkpointing:
            ensure_grad = torch.zeros(1, device=image_features.device).requires_grad_(True)
            return checkpoint.checkpoint(
                self.gen_weight, image_features, iters, weight, ensure_grad,
                use_reentrant=False,
            )
        return self.gen_weight(image_features, iters, weight)
