"""Downstream multiple-choice scoring (spec 21.3).

The batched scorer is an optimization of the plain one, so the contract is that
it changes speed and nothing else. These tests pin that down, and pin the right
-padding assumption it rests on.
"""

from __future__ import annotations

import json

import pytest
import torch

from kkoma.evaluation.downstream import (
    MultipleChoiceExample,
    continuation_logprob,
    evaluate_task,
    load_examples,
    score_choices,
)
from tests.conftest import tiny_config

from kkoma.model.model import KkomaModel


@pytest.fixture
def model(tiny_tokenizer) -> KkomaModel:
    torch.manual_seed(0)
    m = KkomaModel(tiny_config(vocab_size=len(tiny_tokenizer), context_length=128))
    m.eval()
    return m


@pytest.fixture
def examples() -> list[MultipleChoiceExample]:
    return [
        MultipleChoiceExample(
            context="Artificial intelligence can",
            choices=[" do many things.", " 처음부터 직접 학습되었습니다."],
            label=0,
        ),
        # Deliberately ragged: differing choice counts and lengths are what
        # padding has to survive (ARC-Easy ships 3-, 4- and 5-choice items).
        MultipleChoiceExample(
            context="이 모델은",
            choices=[" 처음부터 직접 학습되었습니다.", " scratch.", " a much longer continuation here."],
            label=2,
        ),
        MultipleChoiceExample(
            context="The model is trained from scratch. Kkoma-LLM은 영어와",
            choices=[" 한국어를 모두 처리합니다.", " do"],
            label=1,
        ),
    ]


def test_batched_matches_unbatched(model, tiny_tokenizer, examples, device):
    """The batched scorer must reproduce the one-pair-at-a-time scores."""

    got = score_choices(model, tiny_tokenizer, examples, device, batch_size=4)

    for ex, row in zip(examples, got):
        for ci, choice in enumerate(ex.choices):
            lp, n = continuation_logprob(model, tiny_tokenizer, ex.context, choice, device)
            assert row[ci] == pytest.approx(lp / n, abs=1e-4)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 64])
def test_scores_independent_of_batch_size(model, tiny_tokenizer, examples, device, batch_size):
    """Batch size only changes how much padding rides along; scores must not move."""

    ref = score_choices(model, tiny_tokenizer, examples, device, batch_size=1)
    got = score_choices(model, tiny_tokenizer, examples, device, batch_size=batch_size)
    for a, b in zip(ref, got):
        assert a == pytest.approx(b, abs=1e-4)


def test_right_padding_does_not_leak(model, tiny_tokenizer, device):
    """A long neighbour in the batch must not change a short item's score.

    This is the assumption that makes batching legal at all: causal attention
    means trailing pad is unreachable from real positions. If someone adds a
    non-causal path or left padding, this fails.
    """

    short = MultipleChoiceExample(context="이 모델은", choices=[" scratch."], label=0)
    long = MultipleChoiceExample(
        context="The model is trained from scratch. Artificial intelligence can do many things.",
        choices=[" 처음부터 직접 학습되었습니다. Kkoma-LLM은 영어와 한국어를 모두 처리합니다."],
        label=0,
    )

    alone = score_choices(model, tiny_tokenizer, [short], device, batch_size=8)[0][0]
    together = score_choices(model, tiny_tokenizer, [short, long], device, batch_size=8)[0][0]
    assert alone == pytest.approx(together, abs=1e-4)


def test_margin_max_sign_tracks_correctness(model, tiny_tokenizer, examples, device):
    """margin_max > 0 must mean exactly "the gold choice ranked first"."""

    scores = score_choices(model, tiny_tokenizer, examples, device, batch_size=4)
    n_correct = sum(row.index(max(row)) == ex.label for ex, row in zip(examples, scores))

    res = evaluate_task(model, tiny_tokenizer, examples, device, batch_size=4)
    assert res["n"] == len(examples)
    assert res["acc_norm"] == pytest.approx(n_correct / len(examples))

    for ex, row in zip(examples, scores):
        gold = row[ex.label]
        margin = gold - max(s for i, s in enumerate(row) if i != ex.label)
        assert (margin > 0) == (row.index(max(row)) == ex.label)


def test_reduce_fn_called_once_even_when_empty(model, tiny_tokenizer, device):
    """A rank with no examples still has to enter the collective, or DDP hangs."""

    calls = []

    def fake_reduce(t):
        calls.append(t.clone())
        return t

    res = evaluate_task(model, tiny_tokenizer, [], device, reduce_fn=fake_reduce)
    assert len(calls) == 1
    assert res["n"] == 0


def _trainer_with_downstream(tiny_tokenizer, examples, **cfg_overrides):
    """A Trainer wired to one downstream task, recording what it logs."""

    from kkoma.config import RunConfig
    from kkoma.training.distributed import DistInfo
    from kkoma.training.optimizer import build_optimizer
    from kkoma.training.scheduler import build_scheduler
    from kkoma.training.trainer import Logger, Trainer

    torch.manual_seed(0)
    cfg = RunConfig()
    cfg.model = tiny_config(vocab_size=len(tiny_tokenizer), context_length=128)
    cfg.training.precision = "fp32"
    cfg.logging.backend = "none"
    for k, v in cfg_overrides.items():
        setattr(cfg.training, k, v)

    model = KkomaModel(cfg.model)
    opt = build_optimizer(model, cfg.optimizer)
    dist = DistInfo()
    trainer = Trainer(
        config=cfg, model=model, raw_model=model, optimizer=opt,
        scheduler=build_scheduler(opt, cfg.scheduler, total_steps=10),
        train_loader=[], dist=dist, device=torch.device("cpu"),
        val_loaders={}, logger=Logger(cfg, dist),
        tokenizer=tiny_tokenizer,
        downstream_tasks=[("hellaswag", "en", examples)],
    )
    logged = []
    trainer.logger.log = lambda metrics, step: logged.append((step, metrics))
    return trainer, logged


def test_downstream_logs_expected_metrics(tiny_tokenizer, examples):
    trainer, logged = _trainer_with_downstream(tiny_tokenizer, examples)
    trainer._evaluate_downstream(0)

    assert len(logged) == 1
    step, metrics = logged[0]
    assert step == 0
    assert set(metrics) == {
        "downstream/hellaswag_acc_norm",
        "downstream/hellaswag_margin_max",
        "downstream/hellaswag_margin_mean",
        "downstream/en_avg",
        "downstream/overall_avg",
    }
    # One en task: both aggregates collapse onto it.
    assert metrics["downstream/en_avg"] == metrics["downstream/hellaswag_acc_norm"]
    assert metrics["downstream/overall_avg"] == metrics["downstream/hellaswag_acc_norm"]
    # No ko task configured, so no ko_avg is invented.
    assert "downstream/ko_avg" not in metrics


def test_downstream_disabled_by_flag(tiny_tokenizer, examples):
    trainer, logged = _trainer_with_downstream(tiny_tokenizer, examples)
    trainer.config.evaluation.downstream_enabled = False
    trainer._evaluate_downstream(0)
    assert logged == []


def test_downstream_evaluated_once_per_step(tiny_tokenizer, examples):
    """The final eval can land on an interval boundary; it must not double-log."""

    trainer, logged = _trainer_with_downstream(tiny_tokenizer, examples)
    trainer._evaluate_downstream(5000)
    trainer._evaluate_downstream(5000)
    assert len(logged) == 1


def test_downstream_does_not_pick_best_checkpoint(tiny_tokenizer, examples):
    """Accuracy over 500 questions is too noisy to select on; loss decides."""

    trainer, _ = _trainer_with_downstream(tiny_tokenizer, examples)
    before = trainer.state.best_val_loss
    trainer._evaluate_downstream(0)
    assert trainer.state.best_val_loss == before


def test_train_loop_baseline_and_cadence(tiny_tokenizer, examples, monkeypatch):
    """Step 0 baseline, then every downstream_interval, then a final one.

    The step-0 point is what a later rise is measured against, and the final
    step must not be logged twice when it lands on an interval boundary.
    """

    trainer, logged = _trainer_with_downstream(
        tiny_tokenizer,
        examples,
        micro_batch_size=1,
        grad_accum_steps=1,
        max_tokens=128 * 4,  # context_length 128 -> 4 steps
        eval_interval=10_000,  # keep LM validation out of the way
        downstream_interval=2,
        log_interval=10_000,
    )
    assert trainer.max_steps == 4

    ids = torch.randint(0, len(tiny_tokenizer), (1, 128))
    trainer.train_loader = [{"input_ids": ids, "labels": ids.clone()} for _ in range(8)]
    monkeypatch.setattr(trainer, "_save", lambda step, tag: None)
    trainer.train()

    steps = [s for s, m in logged if "downstream/overall_avg" in m]
    assert steps == [0, 2, 4]


def test_resume_skips_step_0_baseline(tiny_tokenizer, examples, monkeypatch):
    """A resumed run is mid-training; step 0 is not its baseline to take."""

    trainer, logged = _trainer_with_downstream(
        tiny_tokenizer,
        examples,
        micro_batch_size=1,
        grad_accum_steps=1,
        max_tokens=128 * 4,
        eval_interval=10_000,
        downstream_interval=2,
        log_interval=10_000,
    )
    trainer.state.global_step = 3  # as if resumed
    ids = torch.randint(0, len(tiny_tokenizer), (1, 128))
    trainer.train_loader = [{"input_ids": ids, "labels": ids.clone()} for _ in range(8)]
    monkeypatch.setattr(trainer, "_save", lambda step, tag: None)
    trainer.train()

    steps = [s for s, m in logged if "downstream/overall_avg" in m]
    assert 0 not in steps
    assert steps == [4]


def test_load_examples_roundtrip(tmp_path):
    path = tmp_path / "task.jsonl"
    rows = [
        {"id": "a", "context": "Q1", "choices": [" x", " y"], "label": 1},
        {"id": "b", "context": "Q2", "choices": [" p", " q", " r"], "label": 0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    got = load_examples(str(path))
    assert [e.context for e in got] == ["Q1", "Q2"]
    assert [e.label for e in got] == [1, 0]
    assert [len(e.choices) for e in got] == [2, 3]
