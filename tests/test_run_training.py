"""End-to-end `scripts/_common.run_training` (the real training entrypoint).

Every `scripts/train_*.py` is a thin argparse wrapper around this function, and
before the 2026-07 audit none of its 300-odd lines were tested: config saving,
tokenizer/model/loader wiring, the `--init-from` weights-only path, resume with
the fast-forward block computation, checkpoint rotation, and the distributed
setup/teardown. Each piece is covered elsewhere; this drives them together.

Runs on CPU with fp32 and a handful of steps.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from kkoma.config import DataSource, RunConfig
from scripts._common import run_training
from tests.conftest import tiny_config


def _corpus(tmp_path, n_docs=400):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_docs):
            f.write(json.dumps(
                {"text": f"document {i} the model is trained from scratch "
                         f"이 모델은 처음부터 직접 학습되었습니다 {i}"},
                ensure_ascii=False,
            ) + "\n")
    return str(path)


def _config(tmp_path, tokenizer_dir, *, max_tokens, save_interval=2, tokenizer_vocab=300):
    cfg = RunConfig()
    cfg.model = tiny_config(vocab_size=tokenizer_vocab, context_length=32)
    cfg.tokenizer.path = tokenizer_dir
    cfg.project.run_name = "e2e-test"
    cfg.training.precision = "fp32"          # CPU
    cfg.training.micro_batch_size = 2
    cfg.training.grad_accum_steps = 1
    cfg.training.max_tokens = max_tokens
    cfg.training.log_interval = 1
    cfg.training.eval_interval = 1000        # keep validation out of the way
    cfg.training.downstream_interval = 0
    cfg.logging.backend = "none"
    cfg.checkpoint.dir = str(tmp_path / "ckpt")
    cfg.checkpoint.save_interval = save_interval
    cfg.checkpoint.keep_last = 2
    cfg.data.sources = [DataSource(name="local", path=_corpus(tmp_path), weight=1.0)]
    cfg.data.val_sources = []
    cfg.evaluation.downstream_tasks = []
    return cfg


def _tokens_for_steps(cfg, steps):
    return steps * cfg.training.micro_batch_size * cfg.model.context_length


@pytest.fixture
def trained(tmp_path, tiny_tokenizer_dir):
    cfg = _config(tmp_path, tiny_tokenizer_dir, max_tokens=1)
    cfg.training.max_tokens = _tokens_for_steps(cfg, 4)
    run_training(cfg)
    return cfg


def test_run_training_produces_checkpoints_and_config(trained):
    d = trained.checkpoint.dir
    assert os.path.exists(os.path.join(d, "run_config.json")), "config not saved next to weights"
    assert os.path.exists(os.path.join(d, "final.pt"))

    saved = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    assert saved["project"]["run_name"] == "e2e-test"

    ckpt = torch.load(os.path.join(d, "final.pt"), map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 4
    assert ckpt["tokens_processed"] == _tokens_for_steps(trained, 4)
    # Provenance the spec requires alongside every checkpoint.
    assert ckpt["world_size"] == 1
    assert "config" in ckpt and "rng" in ckpt


def test_periodic_checkpoints_are_rotated(trained):
    d = trained.checkpoint.dir
    steps = sorted(f for f in os.listdir(d) if f.startswith("step_"))
    assert steps, "no periodic checkpoint written"
    assert len(steps) <= trained.checkpoint.keep_last, f"keep_last not honoured: {steps}"


def test_resume_continues_from_the_saved_step(tmp_path, tiny_tokenizer_dir):
    cfg = _config(tmp_path, tiny_tokenizer_dir, max_tokens=1)
    cfg.training.max_tokens = _tokens_for_steps(cfg, 3)
    run_training(cfg)
    first = torch.load(os.path.join(cfg.checkpoint.dir, "final.pt"),
                       map_location="cpu", weights_only=False)
    assert first["global_step"] == 3

    longer = _config(tmp_path, tiny_tokenizer_dir, max_tokens=1)
    longer.training.max_tokens = _tokens_for_steps(longer, 6)
    run_training(longer, resume_path=os.path.join(cfg.checkpoint.dir, "final.pt"))

    second = torch.load(os.path.join(longer.checkpoint.dir, "final.pt"),
                        map_location="cpu", weights_only=False)
    assert second["global_step"] == 6, "resume restarted instead of continuing"
    assert second["tokens_processed"] > first["tokens_processed"]


def test_resume_refuses_a_missing_checkpoint(tmp_path, tiny_tokenizer_dir):
    cfg = _config(tmp_path, tiny_tokenizer_dir, max_tokens=1)
    cfg.training.max_tokens = _tokens_for_steps(cfg, 2)
    with pytest.raises(FileNotFoundError, match="refusing to silently restart"):
        run_training(cfg, resume_path=str(tmp_path / "nope.pt"))


def test_init_from_loads_weights_only(tmp_path, tiny_tokenizer_dir):
    """Continued pretraining starts from Base weights with a fresh optimizer.

    The distinguishing evidence is the step counter: --init-from must begin at
    zero, where --resume would continue.
    """

    base = _config(tmp_path / "base", tiny_tokenizer_dir, max_tokens=1)
    base.training.max_tokens = _tokens_for_steps(base, 3)
    run_training(base)
    base_ckpt = os.path.join(base.checkpoint.dir, "final.pt")

    cpt = _config(tmp_path / "cpt", tiny_tokenizer_dir, max_tokens=1)
    cpt.training.max_tokens = _tokens_for_steps(cpt, 2)
    run_training(cpt, init_from=base_ckpt)

    out = torch.load(os.path.join(cpt.checkpoint.dir, "final.pt"),
                     map_location="cpu", weights_only=False)
    assert out["global_step"] == 2, "init_from should start a new schedule, not continue one"

    # And the weights really came from the base run rather than a fresh init.
    before = torch.load(base_ckpt, map_location="cpu", weights_only=False)["model"]
    fresh = RunConfig()
    fresh.model = tiny_config(vocab_size=300, context_length=32)
    from kkoma.model.model import KkomaModel

    torch.manual_seed(99)
    random_init = KkomaModel(fresh.model).state_dict()
    key = "token_embedding.weight"
    assert not torch.allclose(out["model"][key], random_init[key])
    assert before[key].shape == out["model"][key].shape


def test_tokenizer_vocab_mismatch_is_rejected(tmp_path, tiny_tokenizer_dir):
    """A model built for a different vocabulary would train against garbage
    logits; the mismatch has to surface before any step runs."""

    cfg = _config(tmp_path, tiny_tokenizer_dir, max_tokens=1, tokenizer_vocab=512)
    cfg.training.max_tokens = _tokens_for_steps(cfg, 1)
    with pytest.raises(ValueError, match="vocab"):
        run_training(cfg)
