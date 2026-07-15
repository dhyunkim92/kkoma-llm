"""Causal self-attention with optional Grouped-Query Attention (GQA).

Uses ``F.scaled_dot_product_attention`` (no hard FlashAttention-2 dependency, per
the V100 target). Supports:
- MHA (n_kv_head == n_head) and GQA (n_kv_head < n_head) via KV head repetition
- RoPE applied to Q/K with a position offset
- incremental decoding through a per-layer KV cache
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kkoma.config import ModelConfig
from kkoma.model.cache import LayerKVCache
from kkoma.model.positional import apply_rotary


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA.

    x: ``[batch, n_kv_heads, seq, head_dim]`` -> ``[batch, n_kv_heads*n_rep, seq, head_dim]``.
    """

    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = config.n_query_groups
        self.use_rope = config.positional_encoding == "rope"
        self.dropout = config.dropout

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.q_proj = nn.Linear(config.d_model, q_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(q_dim, config.d_model, bias=config.bias)
        self.o_proj._is_residual_proj = True

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            assert cos is not None and sin is not None, "RoPE tables required"
            q, k = apply_rotary(q, k, cos, sin)

        # Incremental decoding: append to cache and attend over the full prefix.
        if cache is not None:
            k, v = cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Masking has three cases:
        # - full-sequence forward (q_len == k_len): plain causal mask
        # - single-token decode (q_len == 1): the new token may attend the
        #   whole prefix, no mask needed
        # - chunked forward into a non-empty cache (1 < q_len < k_len): each
        #   chunk token attends all cached positions plus the causal prefix of
        #   the chunk, which needs an explicit mask (is_causal would misalign
        #   and no mask would leak future chunk tokens)
        q_len, k_len = q.shape[2], k.shape[2]
        is_causal = q_len == k_len
        attn_mask = None
        if not is_causal and q_len > 1:
            attn_mask = torch.ones(
                q_len, k_len, dtype=torch.bool, device=q.device
            ).tril(diagonal=k_len - q_len)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn)


__all__ = ["CausalSelfAttention", "repeat_kv"]
