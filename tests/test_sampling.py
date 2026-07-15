"""Sampling filter tests + scheduler/grad-accum wiring regression tests."""

from __future__ import annotations

import torch

from kkoma.config import (
    ModelConfig,
    RunConfig,
    TrainingConfig,
    resolve_grad_accum,
    tokens_per_optimizer_step,
)
from kkoma.generation.sampling import filter_top_k, filter_top_p, sample_next_token


def test_top_p_keeps_crossing_token():
    # probs 0.5, 0.3, 0.15, 0.05 -> cumsum 0.5, 0.8, 0.95, 1.0
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    out = filter_top_p(logits.clone(), 0.7)
    keep = (out[0] != float("-inf")).tolist()
    # keep top token (0.5) and the one that crosses 0.7 (0.3); drop the rest.
    assert keep == [True, True, False, False]


def test_top_p_preserves_original_order():
    logits = torch.tensor([[0.05, 0.5, 0.15, 0.3]]).log()
    out = filter_top_p(logits.clone(), 0.7)
    keep = (out[0] != float("-inf")).tolist()
    assert keep == [False, True, False, True]


def test_top_p_passthrough_when_one():
    logits = torch.randn(2, 10)
    assert torch.equal(filter_top_p(logits.clone(), 1.0), logits)


def test_top_k_keeps_k_tokens():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    out = filter_top_k(logits.clone(), 2)
    assert (out[0] != float("-inf")).sum().item() == 2


def test_fp16_logits_low_temperature_no_overflow():
    """fp16 logits at low temperature must not overflow to inf/NaN (fp32 upcast)."""

    # 300 / 0.01 = 30,000 is fine in fp32 but 300 stored as fp16 divided in
    # fp16 at temperature 0.005 -> 60,000, and 400 -> 80,000 > fp16 max.
    logits = torch.tensor([[400.0, 399.0, 1.0, 0.0]]).half()
    gen = torch.Generator().manual_seed(0)
    token = sample_next_token(logits, temperature=0.005, generator=gen)
    assert token.item() in (0, 1)  # valid draw from the two dominant tokens


def _cfg(grad_accum=None, gbt=262144, micro=4, ctx=1024, max_tokens=2_500_000):
    return RunConfig(
        model=ModelConfig(context_length=ctx),
        training=TrainingConfig(
            global_batch_tokens=gbt,
            micro_batch_size=micro,
            grad_accum_steps=grad_accum,
            max_tokens=max_tokens,
        ),
    )


def test_grad_accum_derived_from_global_batch():
    cfg = _cfg(grad_accum=None, gbt=262144, micro=4, ctx=1024)
    # 262144 / (4*1024*1) = 64
    assert resolve_grad_accum(cfg, world_size=1) == 64


def test_grad_accum_explicit_wins():
    cfg = _cfg(grad_accum=8, gbt=262144)
    assert resolve_grad_accum(cfg, world_size=1) == 8


def test_grad_accum_scales_with_world_size():
    cfg = _cfg(grad_accum=None, gbt=262144, micro=4, ctx=1024)
    assert resolve_grad_accum(cfg, world_size=8) == 8  # 262144/(4*1024*8)


def test_tokens_per_step_and_horizon_consistent():
    cfg = _cfg(grad_accum=None, gbt=262144, micro=4, ctx=1024, max_tokens=2_500_000)
    tps = tokens_per_optimizer_step(cfg, world_size=1)
    assert tps == 262144  # one global batch per optimizer step
    # scheduler horizon = trainer max_steps
    assert cfg.training.max_tokens // tps == 9
