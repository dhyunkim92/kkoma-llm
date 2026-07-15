"""Checkpoint resume tests (spec 18): model/optimizer/step restored exactly."""

from __future__ import annotations

import json

import torch

from kkoma.config import DataSource, OptimizerConfig, RunConfig
from kkoma.model.model import KkomaModel
from kkoma.training.checkpoint import (
    apply_checkpoint,
    collect_provenance,
    load_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from kkoma.training.optimizer import build_optimizer
from tests.conftest import tiny_config


def _train_a_bit(model, opt, steps, seed=0):
    torch.manual_seed(seed)
    model.train()
    for _ in range(steps):
        x = torch.randint(0, model.config.vocab_size, (2, 8))
        opt.zero_grad()
        _, loss = model(x, labels=x)
        loss.backward()
        opt.step()
    return loss.item()


def test_resume_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(0)
    model = KkomaModel(tiny_config())
    opt = build_optimizer(model, OptimizerConfig(learning_rate=1e-3))
    _train_a_bit(model, opt, 3)

    path = str(tmp_path / "step.pt")
    save_checkpoint(
        path, model, opt, None, None,
        global_step=3, tokens_processed=3 * 2 * 8, config_dict={},
    )

    # Fresh model + optimizer, then resume.
    torch.manual_seed(123)
    model2 = KkomaModel(tiny_config())
    opt2 = build_optimizer(model2, OptimizerConfig(learning_rate=1e-3))
    ckpt = load_checkpoint(path, model2, opt2)

    assert ckpt["global_step"] == 3
    assert ckpt["tokens_processed"] == 48
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

    # Continuing training from each should match step-for-step.
    loss_a = _train_a_bit(model, opt, 2, seed=99)
    loss_b = _train_a_bit(model2, opt2, 2, seed=99)
    assert abs(loss_a - loss_b) < 1e-5


def test_read_then_apply_matches_load(tmp_path):
    """read_checkpoint exposes metadata before apply_checkpoint touches state."""

    torch.manual_seed(0)
    model = KkomaModel(tiny_config())
    opt = build_optimizer(model, OptimizerConfig())
    _train_a_bit(model, opt, 1)
    path = str(tmp_path / "c.pt")
    save_checkpoint(
        path, model, opt, None, None,
        global_step=1, tokens_processed=16, config_dict={},
        extra={"wandb_run_id": "run-abc123"},
    )

    ckpt = read_checkpoint(path)
    assert ckpt["wandb_run_id"] == "run-abc123"  # readable pre-apply (spec 23)

    torch.manual_seed(7)
    model2 = KkomaModel(tiny_config())
    apply_checkpoint(ckpt, model2)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_collect_provenance_hashes_tokenizer_and_manifest(tmp_path):
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "data_manifest.json").write_text(
        json.dumps({"sampling_seed": 42}), encoding="utf-8"
    )

    config = RunConfig()
    config.tokenizer.path = str(tok_dir)
    config.data.sources = [DataSource(name="local", path=str(data_dir / "*.jsonl"))]

    prov = collect_provenance(config)
    assert prov["tokenizer_id"]["sha256"]  # tokenizer.json hashed (spec 18.1)
    manifest_path = str(data_dir / "data_manifest.json")
    assert prov["data_manifests"][manifest_path]
    assert "git_commit" in prov and "torch" in prov["env"]


def test_optimizer_state_present_after_save(tmp_path):
    torch.manual_seed(0)
    model = KkomaModel(tiny_config())
    opt = build_optimizer(model, OptimizerConfig())
    _train_a_bit(model, opt, 1)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, model, opt, None, None, global_step=1, tokens_processed=1, config_dict={})
    ckpt = torch.load(path, weights_only=False)
    assert "optimizer" in ckpt and ckpt["optimizer"]["state"]
