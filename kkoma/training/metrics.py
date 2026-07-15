"""Training metrics and FP16 stability checks (spec sections 17 / 23)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch


@dataclass
class Throughput:
    """Exponential / windowed tokens-per-second tracker."""

    _last_time: float = field(default_factory=time.perf_counter)
    _tokens: int = 0

    def update(self, n_tokens: int) -> None:
        self._tokens += n_tokens

    def rate(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._last_time
        tps = self._tokens / elapsed if elapsed > 0 else 0.0
        self._tokens = 0
        self._last_time = now
        return tps


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def is_finite(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss).all().item())


def grad_global_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total)


def gpu_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


__all__ = [
    "Throughput",
    "perplexity",
    "is_finite",
    "grad_global_norm",
    "gpu_memory_stats",
]
