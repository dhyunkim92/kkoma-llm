"""GQA tests (spec 6.4 / 10): head divisibility and KV head repetition."""

from __future__ import annotations

import pytest
import torch

from kkoma.config import ModelConfig
from kkoma.model.attention import CausalSelfAttention, repeat_kv
from kkoma.model.positional import RotaryEmbedding
from tests.conftest import tiny_config


def test_repeat_kv_expands_heads():
    x = torch.randn(2, 2, 5, 16)  # [b, n_kv, seq, head_dim]
    out = repeat_kv(x, n_rep=4)
    assert out.shape == (2, 8, 5, 16)
    # Each query group sees the same KV head it was expanded from.
    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.allclose(out[:, 3], x[:, 0])
    assert torch.allclose(out[:, 4], x[:, 1])


def test_repeat_kv_identity_when_one():
    x = torch.randn(1, 4, 3, 8)
    assert torch.equal(repeat_kv(x, 1), x)


def test_gqa_requires_divisible_heads():
    with pytest.raises(ValueError):
        ModelConfig(n_head=8, n_kv_head=3, d_model=64, head_dim=8)


def test_gqa_query_groups():
    cfg = tiny_config(n_head=8, n_kv_head=2, d_model=128, head_dim=16)
    assert cfg.n_query_groups == 4


def test_mha_and_gqa_forward():
    for n_kv in (4, 2, 1):
        cfg = tiny_config(n_head=4, n_kv_head=n_kv, d_model=64, head_dim=16)
        attn = CausalSelfAttention(cfg)
        rope = RotaryEmbedding(cfg.head_dim, max_seq_len=cfg.context_length)
        cos, sin = rope(7)
        x = torch.randn(2, 7, cfg.d_model)
        assert attn(x, cos=cos, sin=sin).shape == x.shape
