"""RoPE tests (spec 7.2): shape preservation, determinism, offset support."""

from __future__ import annotations

import torch

from kkoma.model.positional import RotaryEmbedding, apply_rotary


def test_rope_shape_preserved():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    cos, sin = rope(seq_len=8)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 2, 8, 16)
    # broadcast across heads
    q2, _ = apply_rotary(q, q, cos, sin)
    assert q2.shape == q.shape


def test_rope_deterministic():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    c1, s1 = rope(seq_len=8)
    c2, s2 = rope(seq_len=8)
    assert torch.allclose(c1, c2)
    assert torch.allclose(s1, s2)


def test_rope_offset_matches_slice():
    """cos/sin at offset k must equal the k-shifted slice of the full table."""

    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    full_cos, full_sin = rope(seq_len=20, offset=0)
    off_cos, off_sin = rope(seq_len=5, offset=10)
    assert torch.allclose(off_cos, full_cos[10:15])
    assert torch.allclose(off_sin, full_sin[10:15])


def test_rope_extends_cache():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    cos, sin = rope(seq_len=32)  # forces a rebuild beyond initial cache
    assert cos.shape == (32, 8)


def test_model_rejects_sequences_past_context_length():
    """RoPE extends its table on demand, so nothing else stops extrapolation.

    Before this guard a forward past context_length ran to completion and
    returned confident nonsense; only generate() checked the bound.
    """

    import pytest

    from kkoma.model.model import KkomaModel
    from tests.conftest import tiny_config

    cfg = tiny_config(context_length=32)
    model = KkomaModel(cfg).eval()
    ok = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    model(ok)  # exactly at the limit is fine

    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 1))
    with pytest.raises(ValueError, match="exceeds context_length"):
        model(too_long)


def test_rotation_preserves_norm():
    """RoPE is a rotation, so per-position vector norms are preserved."""

    rope = RotaryEmbedding(head_dim=16, max_seq_len=16)
    cos, sin = rope(seq_len=4)
    q = torch.randn(1, 1, 4, 16)
    q_rot, _ = apply_rotary(q, q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
