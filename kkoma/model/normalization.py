"""Normalization layers.

LayerNorm is used in the Architecture Study baseline; RMSNorm is the modern
candidate used by the final Kkoma Base structure. RMSNorm accumulates in FP32
and casts back to the input dtype for stability under FP16 training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kkoma.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x.to(input_dtype))


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (bias controlled by config)."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )


def make_norm(config: ModelConfig) -> nn.Module:
    """Factory selecting the normalization module from config."""

    if config.norm == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.norm_eps)
    if config.norm == "layernorm":
        return LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
    raise ValueError(f"unknown norm: {config.norm}")


__all__ = ["RMSNorm", "LayerNorm", "make_norm"]
