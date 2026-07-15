"""Positional encodings: rotary (RoPE) and learned absolute embeddings.

RoPE is applied to Q and K. The cos/sin tables are precomputed up to the
training context length and extended on demand. A ``position offset`` is
supported so that incremental KV-cache decoding rotates new tokens by their
absolute position.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kkoma.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotary position embedding cache.

    Stores inverse frequencies as a buffer and lazily (re)builds the cos/sin
    tables when a longer sequence than currently cached is requested.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 1024):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, offset: int = 0, device=None, dtype=torch.float32):
        """Return (cos, sin) slices covering positions [offset, offset+seq_len)."""

        need = offset + seq_len
        if need > self._cached_len or self.cos_cached.device != self.inv_freq.device:
            self._build_cache(max(need, self._cached_len * 2 or need))
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors.

    q, k: ``[batch, n_heads, seq, head_dim]``.
    cos, sin: ``[seq, head_dim]`` -> broadcast over batch and heads.
    """

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class LearnedPositionalEmbedding(nn.Module):
    """Absolute learned positional embedding, used only in the baseline."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, offset: int = 0, device=None) -> torch.Tensor:
        positions = torch.arange(offset, offset + seq_len, device=device)
        return self.embedding(positions)  # [seq, d_model]


def build_rope(config: ModelConfig) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_dim=config.head_dim,
        base=config.rope_theta,
        max_seq_len=config.context_length,
    )


__all__ = [
    "RotaryEmbedding",
    "LearnedPositionalEmbedding",
    "apply_rotary",
    "rotate_half",
    "build_rope",
]
