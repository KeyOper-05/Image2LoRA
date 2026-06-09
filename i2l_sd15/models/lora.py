"""Standard LoRA for SD 1.5 UNet attention layers."""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
UNET_TARGET_LINEAR_NAMES = ["to_q", "to_k", "to_v", "to_out.0"]
LORA_PREFIX_UNET = "lora_unet"


@dataclass
class LoRALayerSpec:
    name: str
    in_features: int
    out_features: int
    rank: int

    @property
    def param_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)


class LoRAModule(torch.nn.Module):
    """Classic LoRA with dynamically assigned weights from the i2L generator."""

    def __init__(self, lora_name: str, org_module: torch.nn.Module, rank: int, alpha: float, multiplier: float = 1.0):
        super().__init__()
        self.lora_name = lora_name
        self.rank = rank
        self.network_alpha = alpha
        self.multiplier = multiplier
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.org_module = org_module
        self.lora_down = None
        self.lora_up = None

    def set_lora_weights(self, lora_down: torch.Tensor, lora_up: torch.Tensor):
        self.lora_down = lora_down
        self.lora_up = lora_up

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.org_forward(hidden_states)
        if self.lora_down is None or self.lora_up is None:
            return out

        orig_dtype = hidden_states.dtype
        dtype = self.lora_down.dtype
        actual_rank = self.lora_down.shape[1] if self.lora_down.dim() == 3 else self.lora_down.shape[0]
        scale = self.network_alpha / actual_rank

        if self.lora_down.dim() == 2:
            mid = F.linear(hidden_states.to(dtype), self.lora_down)
            delta = F.linear(mid, self.lora_up) * scale
        else:
            mid = torch.einsum("b ... i, b r i -> b ... r", hidden_states.to(dtype), self.lora_down)
            delta = torch.einsum("b o r, b ... r -> b ... o", self.lora_up, mid) * scale

        return out + (delta * self.multiplier).to(orig_dtype)


class LoRANetwork(torch.nn.Module):
    def __init__(self, unet, rank: int = 4, alpha: float = 4.0, multiplier: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.multiplier = multiplier
        self.unet_loras: List[LoRAModule] = []
        self.layer_specs: List[LoRALayerSpec] = []

        for name, module in unet.named_modules():
            if module.__class__.__name__ not in UNET_TARGET_REPLACE_MODULE:
                continue
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ not in ["Linear", "LoRACompatibleLinear"]:
                    continue
                if child_name.split(".")[-1] not in UNET_TARGET_LINEAR_NAMES:
                    continue
                lora_name = f"{LORA_PREFIX_UNET}_{name}_{child_name}".replace(".", "_")
                lora = LoRAModule(lora_name, child_module, rank=rank, alpha=alpha, multiplier=multiplier)
                self.unet_loras.append(lora)
                self.layer_specs.append(
                    LoRALayerSpec(lora_name, lora.in_features, lora.out_features, rank)
                )
                self.add_module(lora_name, lora)

        total = sum(s.param_count for s in self.layer_specs)
        print(f"i2L LoRA targets: {len(self.unet_loras)} layers, rank={rank}, total_lora_dim={total}")

    def apply_to(self, unet):
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def set_lora_weights(self, weight_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        for (down, up), lora in zip(weight_pairs, self.unet_loras):
            lora.set_lora_weights(down, up)


def collect_layer_specs(unet, rank: int = 4) -> List[LoRALayerSpec]:
    specs = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ not in UNET_TARGET_REPLACE_MODULE:
            continue
        for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ not in ["Linear", "LoRACompatibleLinear"]:
                continue
            if child_name.split(".")[-1] not in UNET_TARGET_LINEAR_NAMES:
                continue
            lora_name = f"{LORA_PREFIX_UNET}_{name}_{child_name}".replace(".", "_")
            specs.append(LoRALayerSpec(lora_name, child_module.in_features, child_module.out_features, rank))
    return specs


def create_network(unet, rank: int = 4, alpha: float = 4.0, multiplier: float = 1.0) -> LoRANetwork:
    return LoRANetwork(unet, rank=rank, alpha=alpha, multiplier=multiplier)
