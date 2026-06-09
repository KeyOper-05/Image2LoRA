"""Image-to-LoRA generator: DINOv2 features -> per-layer 2-layer MLP -> LoRA weights.

Architecture from Qwen-Image-i2L / 知乎文章:
  - 不用 Transformer 统一处理所有 LoRA 张量
  - 每个 UNet LoRA 层用独立的双层全连接网络生成权重
  - 图像编码器使用 DINOv2（冻结）
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .lora import LoRALayerSpec


class LoRALayerMLP(nn.Module):
    """双层 FC: image_feat -> LoRA down/up 权重。"""

    def __init__(self, feat_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Image2LoRAGenerator(nn.Module):
    """
    多个双层全连接层，每个对应一个 LoRA 张量组 (lora_down + lora_up)。
    输入: DINOv2 全局特征 (B, feat_dim)
    输出: List[(lora_down, lora_up)]，长度 = UNet LoRA 层数
    """

    def __init__(
        self,
        layer_specs: List[LoRALayerSpec],
        feat_dim: int = 768,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.layer_specs = layer_specs
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.generators = nn.ModuleList([
            LoRALayerMLP(feat_dim, hidden_dim, spec.param_count)
            for spec in layer_specs
        ])
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Image2LoRAGenerator: {len(layer_specs)} MLPs, hidden={hidden_dim}, params={n_params/1e6:.2f}M")

    @staticmethod
    def pool_image_features(features: torch.Tensor) -> torch.Tensor:
        """DINOv2 patch tokens (B, L, D) -> 全局特征 (B, D)。"""
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    def forward(self, image_features: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.pool_image_features(image_features)
        outputs = []
        for gen, spec in zip(self.generators, self.layer_specs):
            flat = gen(feat)
            down = flat[:, : spec.rank * spec.in_features].view(-1, spec.rank, spec.in_features)
            up = flat[:, spec.rank * spec.in_features :].view(-1, spec.out_features, spec.rank)
            outputs.append((down, up))
        return outputs

    def forward_multi_images(self, image_features_list: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        多参考图：每张图生成一组 LoRA，沿 rank 维拼接（与 i2L 文章一致）。
        image_features_list: N x (1, L, D) 或 (1, D)
        返回 batch=1 的 LoRA 权重。
        """
        if len(image_features_list) == 1:
            return self.forward(image_features_list[0])

        per_image = [self.forward(f) for f in image_features_list]
        merged = []
        for layer_idx in range(len(self.layer_specs)):
            downs = [per_image[i][layer_idx][0] for i in range(len(per_image))]
            ups = [per_image[i][layer_idx][1] for i in range(len(per_image))]
            merged.append((
                torch.cat(downs, dim=1),  # (1, rank*N, in)
                torch.cat(ups, dim=2),    # (1, out, rank*N)
            ))
        return merged
