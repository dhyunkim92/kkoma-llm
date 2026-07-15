"""AdamW with decoupled weight-decay parameter groups.

Weight decay applies to 2D matrices (linear / embedding weights); norms and
biases are excluded. With tied embeddings the shared parameter appears once in
the optimizer's parameter list (deduplicated by id).
"""

from __future__ import annotations

import torch

from kkoma.config import OptimizerConfig


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if config.name.lower() != "adamw":
        raise ValueError(f"unsupported optimizer: {config.name}")

    # Use fused AdamW when available (CUDA) for throughput.
    extra = {}
    if torch.cuda.is_available():
        extra["fused"] = True

    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        **extra,
    )


__all__ = ["build_optimizer"]
