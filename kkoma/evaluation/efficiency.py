"""Throughput, memory, and FLOPs estimation (spec sections 21.2 / 6.7).

Provides a rough analytical FLOPs estimate (the common ``6 * N * D`` rule for a
forward+backward training step) and a small benchmarking helper to measure
tokens/second, step time, and peak GPU memory.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from kkoma.config import ModelConfig
from kkoma.model.model import KkomaModel


def estimate_flops_per_token(model: KkomaModel) -> float:
    """Approximate training FLOPs per token as 6 * non_embedding_params.

    This is the standard Kaplan/Chinchilla approximation (2 for forward, 4 for
    backward) over the non-embedding parameter count.
    """

    n = model.num_parameters(non_embedding=True)
    return 6.0 * n


def attention_kv_cache_bytes(config: ModelConfig, batch: int, seq_len: int, dtype_bytes: int = 2) -> int:
    """Bytes used by the KV cache: 2 (K+V) * layers * kv_heads * seq * head_dim."""

    return (
        2
        * config.n_layer
        * config.n_kv_head
        * config.head_dim
        * batch
        * seq_len
        * dtype_bytes
    )


@torch.no_grad()
def benchmark_forward(
    model: KkomaModel,
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 1024,
    steps: int = 10,
    warmup: int = 3,
) -> dict:
    """Measure forward tokens/second and peak memory on random inputs."""

    model.eval()
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = batch_size * seq_len * steps
    mem = (
        torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    )
    return {
        "tokens_per_second": tokens / elapsed,
        "step_time_ms": 1000 * elapsed / steps,
        "peak_allocated_mb": mem,
        "parameters": model.num_parameters(),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "flops_per_token": estimate_flops_per_token(model),
    }


__all__ = [
    "estimate_flops_per_token",
    "attention_kv_cache_bytes",
    "benchmark_forward",
]
