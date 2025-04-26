from typing import Tuple

import torch
import torch.nn as nn

from .registry import register_adapter


@register_adapter("lora_adapter")
class LoRAAdapter(nn.Module):
    """
    Insert LoRA adapter into the model.

    Applies LoRA to transformer blocks.
    """

    def __init__(self, r: int, input_dim: int, output_dim: int):
        super().__init__()
        self.w_a = nn.Linear(input_dim, r, bias=False)
        self.w_b = nn.Linear(r, output_dim, bias=False)

    def forward(self, x):
        x = self.w_b(self.w_a(x))
        return x


@register_adapter("residual_adapter")
class AdapterBlock(nn.Module):
    def __init__(self, blk: nn.Module):
        super().__init__()
        dim = blk.attn.qkv.in_features
        self.adapters = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, dim), nn.GELU()
        )

    def forward(self, x):
        x = self.adapters(x)
        return x


def adapter_residual(
    adapter_module: nn.Module, module: nn.Module, inputs: Tuple[torch.Tensor]
):
    """
    Apply adapter to input and add residual connection.
    """

    # Inputs is a tuple; extract the first element
    x = inputs[0]
    # Apply the adapter module
    modified_input = adapter_module(x) + x
    return (modified_input,) + inputs[1:]
