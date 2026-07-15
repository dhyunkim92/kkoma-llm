"""Attention tests (spec 22.1): output shape, causal masking, RMSNorm, SwiGLU."""

from __future__ import annotations

import torch

from kkoma.model.attention import CausalSelfAttention
from kkoma.model.feedforward import SwiGLU
from kkoma.model.normalization import RMSNorm
from kkoma.model.positional import RotaryEmbedding
from tests.conftest import tiny_config


def test_attention_output_shape():
    cfg = tiny_config()
    attn = CausalSelfAttention(cfg)
    rope = RotaryEmbedding(cfg.head_dim, max_seq_len=cfg.context_length)
    x = torch.randn(2, 10, cfg.d_model)
    cos, sin = rope(10)
    out = attn(x, cos=cos, sin=sin)
    assert out.shape == x.shape


def test_attention_is_causal():
    """A change at position t must not affect outputs at positions < t."""

    cfg = tiny_config(dropout=0.0)
    attn = CausalSelfAttention(cfg).eval()
    rope = RotaryEmbedding(cfg.head_dim, max_seq_len=cfg.context_length)
    cos, sin = rope(6)
    x = torch.randn(1, 6, cfg.d_model)
    out1 = attn(x, cos=cos, sin=sin)

    x2 = x.clone()
    x2[:, 5, :] += 10.0  # perturb the last position only
    out2 = attn(x2, cos=cos, sin=sin)

    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5)
    assert not torch.allclose(out1[:, 5], out2[:, 5], atol=1e-5)


def test_rmsnorm_numerics():
    norm = RMSNorm(8, eps=1e-5)
    x = torch.randn(4, 8)
    out = norm(x)
    # With unit scale, RMS of the output is ~1.
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_shape():
    ffn = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 5, 64)
    assert ffn(x).shape == x.shape
