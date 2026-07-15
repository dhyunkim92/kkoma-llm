"""KV cache for autoregressive generation.

A ``LayerKVCache`` holds the running K/V for one attention layer; ``KVCache``
bundles one per layer plus the current sequence length used as the RoPE
position offset. Caches store GQA-shaped tensors:
``[batch, n_kv_heads, cached_seq, head_dim]``.
"""

from __future__ import annotations

from typing import Optional

import torch


class LayerKVCache:
    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V along the sequence dim and return the full cache."""

        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class KVCache:
    """A collection of per-layer caches plus a shared position offset."""

    def __init__(self, n_layer: int, max_seq_len: Optional[int] = None):
        self.layers = [LayerKVCache() for _ in range(n_layer)]
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def check_capacity(self, new_tokens: int) -> None:
        if self.max_seq_len is not None and self.seq_len + new_tokens > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: have {self.seq_len}, adding {new_tokens}, "
                f"limit {self.max_seq_len}"
            )

    def reset(self) -> None:
        for layer in self.layers:
            layer.k = None
            layer.v = None


__all__ = ["LayerKVCache", "KVCache"]
