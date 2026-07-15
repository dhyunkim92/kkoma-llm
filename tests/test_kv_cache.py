"""KV cache tests (spec 20.2): cached vs uncached greedy decoding must match."""

from __future__ import annotations

import pytest
import torch

from kkoma.generation.generate import GenerationConfig, generate
from kkoma.model.model import KkomaModel
from tests.conftest import tiny_config


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_cache_equivalence_greedy(n_kv_head):
    torch.manual_seed(0)
    cfg = tiny_config(n_head=4, n_kv_head=n_kv_head, d_model=64, head_dim=16)
    model = KkomaModel(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    base = GenerationConfig(max_new_tokens=15, temperature=0.0)
    with_cache = generate(model, prompt, GenerationConfig(**{**base.__dict__, "use_cache": True}))
    no_cache = generate(model, prompt, GenerationConfig(**{**base.__dict__, "use_cache": False}))
    assert torch.equal(with_cache, no_cache)


def test_cache_position_offset_logits():
    """Cached single-step logits equal a full-sequence forward at the same pos."""

    torch.manual_seed(0)
    cfg = tiny_config()
    model = KkomaModel(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 5))

    full_logits, _ = model(seq)

    cache = model.new_cache()
    step_logits = None
    for t in range(seq.shape[1]):
        step_logits, _ = model(seq[:, t : t + 1], cache=cache)
    # Last cached step predicts the same distribution as the full forward's last pos.
    assert torch.allclose(step_logits[:, -1], full_logits[:, -1], atol=1e-4)


@pytest.mark.parametrize("chunks", [(3, 2), (2, 2, 1), (1, 4)])
def test_chunked_prefill_matches_full_forward(chunks):
    """Multi-token chunks into a non-empty cache must stay causally masked."""

    torch.manual_seed(0)
    cfg = tiny_config()
    model = KkomaModel(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, sum(chunks)))

    full_logits, _ = model(seq)

    cache = model.new_cache()
    pieces, start = [], 0
    for size in chunks:
        logits, _ = model(seq[:, start : start + size], cache=cache)
        pieces.append(logits)
        start += size
    chunked_logits = torch.cat(pieces, dim=1)

    # Every position, not just the last: a missing mask would leak future
    # tokens within a chunk and change the earlier positions' logits.
    assert torch.allclose(chunked_logits, full_logits, atol=1e-4)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_rejects_over_context_request(use_cache):
    cfg = tiny_config(context_length=16)
    model = KkomaModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    with pytest.raises(ValueError, match="context_length"):
        generate(model, prompt, GenerationConfig(max_new_tokens=7, use_cache=use_cache))


def test_cache_capacity_error():
    cfg = tiny_config(context_length=8)
    model = KkomaModel(cfg).eval()
    cache = model.new_cache(max_seq_len=4)
    with pytest.raises(ValueError):
        model(torch.randint(0, cfg.vocab_size, (1, 5)), cache=cache)
