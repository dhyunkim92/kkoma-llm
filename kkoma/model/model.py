"""The Kkoma decoder-only Transformer.

Pre-Norm blocks, configurable norm / positional / activation, optional GQA,
weight-tied LM head, and KV-cache-aware forward for generation. Kept at a
nanoGPT level of readability: one block class, one model class.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kkoma.config import ModelConfig
from kkoma.model.attention import CausalSelfAttention
from kkoma.model.cache import KVCache
from kkoma.model.feedforward import make_ffn
from kkoma.model.initialization import apply_init
from kkoma.model.normalization import make_norm
from kkoma.model.positional import LearnedPositionalEmbedding, build_rope


class TransformerBlock(nn.Module):
    """Pre-Norm block: x + Attn(Norm(x)), then x + FFN(Norm(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = make_norm(config)
        self.ffn = make_ffn(config)

    def forward(self, x, cos=None, sin=None, cache=None):
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin, cache=cache)
        x = x + self.ffn(self.norm2(x))
        return x


class KkomaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.use_rope = config.positional_encoding == "rope"
        if self.use_rope:
            self.rope = build_rope(config)
            self.pos_embedding = None
        else:
            self.rope = None
            self.pos_embedding = LearnedPositionalEmbedding(
                config.context_length, config.d_model
            )

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm_f = make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Share the SAME Parameter object (verified by tests).
            self.lm_head.weight = self.token_embedding.weight

        apply_init(self, config)

    # ---- core forward -----------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ):
        """Run the model.

        Returns ``(logits, loss)``. ``loss`` is ``None`` when ``labels`` is not
        given. When ``cache`` is provided, ``input_ids`` is the new chunk and
        positions start at the current cache length.
        """

        b, t = input_ids.shape
        offset = cache.seq_len if cache is not None else 0
        if cache is not None:
            cache.check_capacity(t)
        # RoPE grows its table on demand and learned embeddings index straight
        # into theirs, so without this the model happily runs past the length it
        # was trained on and returns confident nonsense. A KVCache built through
        # new_cache() catches this above, but the uncached path (scoring, a
        # plain forward) has no other guard.
        if offset + t > self.config.context_length:
            raise ValueError(
                f"sequence of {offset + t} tokens exceeds context_length "
                f"{self.config.context_length}; positions beyond it were never "
                "trained and extrapolate silently"
            )

        x = self.token_embedding(input_ids)

        cos = sin = None
        if self.use_rope:
            cos, sin = self.rope(t, offset=offset, device=x.device, dtype=x.dtype)
        else:
            x = x + self.pos_embedding(t, offset=offset, device=x.device)
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = block(x, cos=cos, sin=sin, cache=layer_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Standard causal LM shift; ignore_index=-100 for padded/masked.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ---- helpers ----------------------------------------------------------
    def new_cache(self, max_seq_len: Optional[int] = None) -> KVCache:
        if max_seq_len is None:
            max_seq_len = self.config.context_length
        return KVCache(self.config.n_layer, max_seq_len=max_seq_len)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count parameters.

        With weight tying the embedding/LM-head share storage, so a naive sum
        over unique parameters already avoids double counting. ``non_embedding``
        excludes the token embedding (and learned position embedding).
        """

        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            if self.pos_embedding is not None:
                total -= self.pos_embedding.embedding.weight.numel()
        return total

    def parameter_report(self) -> dict:
        return {
            "total": self.num_parameters(),
            "non_embedding": self.num_parameters(non_embedding=True),
            "embedding": self.token_embedding.weight.numel(),
            "tied_embeddings": self.config.tie_word_embeddings,
            "config": self.config.__dict__,
        }


__all__ = ["KkomaModel", "TransformerBlock"]
