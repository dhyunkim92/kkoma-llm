"""Cosine-decay-with-warmup learning-rate schedule (spec section 15.1).

Linear warmup over ``warmup_ratio`` of total steps, then cosine decay to
``min_lr_ratio * peak_lr``. Implemented as a ``LambdaLR`` multiplier so the
optimizer's base LR stays the peak value.
"""

from __future__ import annotations

import math

import torch

from kkoma.config import SchedulerConfig


def cosine_warmup_lambda(total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: SchedulerConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.name.lower() != "cosine":
        raise ValueError(f"unsupported scheduler: {config.name}")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_warmup_lambda(total_steps, config.warmup_ratio, config.min_lr_ratio),
    )


__all__ = ["build_scheduler", "cosine_warmup_lambda"]
