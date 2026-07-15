"""Weight initialization (spec section 12).

- Embedding / Linear weights: Normal(0, 0.02)
- Bias (when used): zeros
- Residual output projections (attention o_proj, FFN down/out): scaled by
  1 / sqrt(2 * n_layer) so deep residual stacks stay well-conditioned.

Output projections are tagged with ``_is_residual_proj = True`` by their
modules; this pass rescales them after the base init.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kkoma.config import ModelConfig


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def apply_init(model: nn.Module, config: ModelConfig) -> None:
    """Initialize a whole model in-place following the spec."""

    std = config.initializer_range
    model.apply(lambda m: init_weights(m, std))

    # Residual scaling for output projections.
    residual_std = std / math.sqrt(2 * config.n_layer)
    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, "_is_residual_proj", False):
            nn.init.normal_(module.weight, mean=0.0, std=residual_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


__all__ = ["init_weights", "apply_init"]
