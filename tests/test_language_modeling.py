"""Validation-loss computation (kkoma/evaluation/language_modeling.py).

This module produces every `val loss` number in `artifacts/evaluation/*.json`
and the leaderboard, and until the 2026-07 audit it had no tests at all. The
contract that matters: the aggregate is TOKEN-weighted, so a language with more
tokens pulls it more, and masked positions are excluded from both the loss and
the token count.
"""

from __future__ import annotations

import math

import pytest
import torch

from kkoma.evaluation.language_modeling import evaluate_bilingual, evaluate_lm_loss
from kkoma.model.model import KkomaModel
from tests.conftest import tiny_config


@pytest.fixture
def model():
    torch.manual_seed(0)
    return KkomaModel(tiny_config()).eval()


def _batches(cfg, n, batch=2, seq=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), generator=g)
        out.append({"input_ids": ids, "labels": ids.clone()})
    return out


def test_reports_predicted_token_count_not_input_tokens(model):
    """Each block of S tokens yields S-1 predictions (the shift drops one)."""

    cfg = model.config
    r = evaluate_lm_loss(model, _batches(cfg, 3, batch=2, seq=16), torch.device("cpu"))
    assert r["tokens"] == 3 * 2 * (16 - 1)


def test_perplexity_matches_exp_loss(model):
    r = evaluate_lm_loss(model, _batches(model.config, 2), torch.device("cpu"))
    assert r["perplexity"] == pytest.approx(math.exp(r["loss"]), rel=1e-6)


def test_max_batches_truncates(model):
    cfg = model.config
    full = evaluate_lm_loss(model, _batches(cfg, 6), torch.device("cpu"))
    part = evaluate_lm_loss(model, _batches(cfg, 6), torch.device("cpu"), max_batches=2)
    assert part["tokens"] == full["tokens"] // 3
    # max_batches=0 means "no cap", not "no batches".
    uncapped = evaluate_lm_loss(model, _batches(cfg, 6), torch.device("cpu"), max_batches=0)
    assert uncapped["tokens"] == full["tokens"]


def test_ignore_index_excluded_from_loss_and_token_count(model):
    """-100 positions must not contribute, or the mean is diluted toward zero."""

    cfg = model.config
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    clean = [{"input_ids": ids, "labels": ids.clone()}]
    masked_labels = ids.clone()
    masked_labels[:, 8:] = -100
    masked = [{"input_ids": ids, "labels": masked_labels}]

    a = evaluate_lm_loss(model, clean, torch.device("cpu"))
    b = evaluate_lm_loss(model, masked, torch.device("cpu"))
    assert a["tokens"] == 2 * 15
    assert b["tokens"] == 2 * 7          # positions 1..7 survive the shift
    assert math.isfinite(b["loss"])


def test_bilingual_aggregate_is_token_weighted(model):
    """`all` must be the token-weighted mean, not the mean of the two losses.

    With equal token counts the two agree, which is why the trainer's unweighted
    average has never diverged from this one — but it is not the same rule, and
    an unequal split is where they part.
    """

    cfg = model.config
    loaders = {
        "en": _batches(cfg, 4, batch=2, seq=16, seed=1),   # 4x more tokens
        "ko": _batches(cfg, 1, batch=2, seq=16, seed=2),
    }
    r = evaluate_bilingual(model, loaders, torch.device("cpu"))

    en, ko = r["en"], r["ko"]
    expected = (en["loss"] * en["tokens"] + ko["loss"] * ko["tokens"]) / (
        en["tokens"] + ko["tokens"]
    )
    assert r["all"]["loss"] == pytest.approx(expected, rel=1e-9)
    assert r["all"]["tokens"] == en["tokens"] + ko["tokens"]
    # The unweighted mean is a different number here; pin that they differ so a
    # future refactor cannot quietly swap one for the other.
    assert r["all"]["loss"] != pytest.approx((en["loss"] + ko["loss"]) / 2, rel=1e-6)


def test_empty_loader_does_not_divide_by_zero(model):
    r = evaluate_lm_loss(model, [], torch.device("cpu"))
    assert r["tokens"] == 0 and math.isfinite(r["loss"])
