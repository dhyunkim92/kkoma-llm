"""Every architecture-study combination must actually run (spec 5, Phase 1).

The ablation compares a modern stack (RMSNorm + RoPE + SwiGLU) against a
baseline (LayerNorm + learned positions + GELU) by flipping one component at a
time. Until the 2026-07 audit `tiny_config()` was the only config any test
built, and it is modern on all three axes — so the *baseline*, the arm every
result is measured against, was the least exercised code in the repo.

These tests instantiate all eight combinations and drive each through the paths
a real run uses: forward, backward, and cached generation.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from kkoma.config import default_ffn_dim
from kkoma.generation.generate import GenerationConfig, generate
from kkoma.model.model import KkomaModel
from tests.conftest import tiny_config

NORMS = ("rmsnorm", "layernorm")
POSITIONS = ("rope", "learned")
ACTIVATIONS = ("swiglu", "gelu")
COMBOS = list(itertools.product(NORMS, POSITIONS, ACTIVATIONS))


def _config(norm, positional, activation):
    cfg = tiny_config()
    cfg.norm = norm
    cfg.positional_encoding = positional
    cfg.activation = activation
    # d_ff is derived from the activation, so recompute rather than inherit the
    # SwiGLU value from tiny_config.
    cfg.d_ff = default_ffn_dim(cfg.d_model, activation)
    return cfg


@pytest.mark.parametrize("norm,positional,activation", COMBOS)
def test_variant_forward_and_backward(norm, positional, activation):
    torch.manual_seed(0)
    cfg = _config(norm, positional, activation)
    model = KkomaModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss), f"{norm}/{positional}/{activation} produced {loss}"

    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert all(torch.isfinite(p.grad).all() for _, p in model.named_parameters())


@pytest.mark.parametrize("norm,positional,activation", COMBOS)
def test_variant_generates_with_kv_cache(norm, positional, activation):
    """Cached decoding advances a position offset, which the learned-embedding
    path indexes directly and the RoPE path slices its table with. Both were
    only ever exercised on RoPE."""

    torch.manual_seed(0)
    cfg = _config(norm, positional, activation)
    model = KkomaModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    gen_cfg = GenerationConfig(max_new_tokens=6, temperature=0.0, use_cache=True, seed=1)
    with torch.no_grad():
        cached = generate(model, prompt, gen_cfg)
        uncached = generate(
            model, prompt,
            GenerationConfig(max_new_tokens=6, temperature=0.0, use_cache=False, seed=1),
        )
    assert cached.shape == (1, 10)
    # Greedy decoding must not depend on whether a cache was used.
    assert torch.equal(cached, uncached), f"{norm}/{positional}/{activation} cache mismatch"


def _ffn_params(d_model: int, activation: str) -> int:
    d_ff = default_ffn_dim(d_model, activation)
    n_proj = 3 if activation == "swiglu" else 2   # SwiGLU adds a gate projection
    return n_proj * d_model * d_ff


# The dimensions the architecture study actually runs at (configs/architecture/).
STUDY_DIMS = (768, 1024)


@pytest.mark.parametrize("d_model", STUDY_DIMS)
def test_swiglu_and_gelu_ffn_params_are_matched_at_study_dims(d_model):
    """The ablation's premise: SwiGLU vs GELU differ in shape, not in size.

    SwiGLU has three projections instead of two, so `default_ffn_dim` targets
    ~8/3*d_model against GELU's 4*d_model. The 128-rounding makes that exact at
    d_model=768 (core_125m) but leaves SwiGLU 3.1% larger at d_model=1024
    (core_350m) — see the dedicated test below. Anything beyond that and an
    'activation' comparison has become a parameter-count comparison.
    """

    gap = abs(_ffn_params(d_model, "swiglu") - _ffn_params(d_model, "gelu"))
    assert gap / _ffn_params(d_model, "gelu") < 0.05


def test_ffn_parity_is_exact_at_768_and_known_off_at_1024():
    """Pin the rounding's actual effect rather than trusting it is negligible.

    `_round_to_multiple(8/3*d, 128)` lands exactly on parity at 768 and
    overshoots at 1024, so the 350M arm gives SwiGLU ~2% more parameters than
    GELU. Small, but the study's whole claim is that one component changed and
    nothing else did, so the number belongs in a test rather than in nobody's
    head. Raising `multiple_of` or pinning `d_ff` explicitly would close it.
    """

    assert _ffn_params(768, "swiglu") == _ffn_params(768, "gelu")

    swiglu, gelu = _ffn_params(1024, "swiglu"), _ffn_params(1024, "gelu")
    assert swiglu > gelu
    assert (swiglu - gelu) / gelu == pytest.approx(0.031, abs=0.002)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("positional", POSITIONS)
def test_activation_swap_does_not_change_anything_else(norm, positional):
    """Only the FFN may differ between the two activations."""

    shapes = {}
    for activation in ACTIVATIONS:
        model = KkomaModel(_config(norm, positional, activation))
        shapes[activation] = {
            n: tuple(p.shape) for n, p in model.named_parameters()
            if "ffn" not in n and "mlp" not in n
        }
    assert shapes["swiglu"] == shapes["gelu"]


def test_layernorm_and_rmsnorm_are_actually_different_modules():
    """Guard against a dispatch that silently falls back to one branch."""

    from kkoma.model.normalization import LayerNorm, RMSNorm

    rms = KkomaModel(_config("rmsnorm", "rope", "swiglu"))
    ln = KkomaModel(_config("layernorm", "rope", "swiglu"))
    assert isinstance(rms.norm_f, RMSNorm)
    assert isinstance(ln.norm_f, LayerNorm)


def test_learned_positions_are_a_parameter_and_rope_is_not():
    rope = KkomaModel(_config("rmsnorm", "rope", "swiglu"))
    learned = KkomaModel(_config("rmsnorm", "learned", "swiglu"))
    names = dict(learned.named_parameters())
    assert any("pos_embedding" in n for n in names), "learned positions must be trainable"
    assert not any("pos_embedding" in n for n in dict(rope.named_parameters()))
