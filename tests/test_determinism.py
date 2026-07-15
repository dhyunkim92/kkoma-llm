"""Determinism tests (spec 22.2): same seed -> same init, forward, sampling."""

from __future__ import annotations

import torch

from kkoma.generation.generate import GenerationConfig, generate
from kkoma.generation.sampling import sample_next_token
from kkoma.model.model import KkomaModel
from tests.conftest import tiny_config


def test_init_deterministic():
    torch.manual_seed(7)
    a = KkomaModel(tiny_config())
    torch.manual_seed(7)
    b = KkomaModel(tiny_config())
    for p1, p2 in zip(a.parameters(), b.parameters()):
        assert torch.equal(p1, p2)


def test_forward_deterministic(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, tiny_model.config.vocab_size, (2, 12))
    out1, _ = tiny_model(x)
    out2, _ = tiny_model(x)
    assert torch.equal(out1, out2)


def test_sampling_seeded_reproducible():
    logits = torch.randn(3, 50)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    t1 = sample_next_token(logits, temperature=1.0, top_k=10, generator=g1)
    t2 = sample_next_token(logits, temperature=1.0, top_k=10, generator=g2)
    assert torch.equal(t1, t2)


def test_generation_seeded_reproducible(tiny_model):
    tiny_model.eval()
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    cfg = GenerationConfig(max_new_tokens=10, temperature=0.9, top_k=20, seed=2024)
    a = generate(tiny_model, prompt, cfg)
    b = generate(tiny_model, prompt, cfg)
    assert torch.equal(a, b)
