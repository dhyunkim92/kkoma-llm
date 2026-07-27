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


# ---------------------------------------------------------------------------
# Resume geometry compatibility (docs/audit-2026-07.md B-4)
# ---------------------------------------------------------------------------


def _ckpt_payload(world_size=2, micro_batch=4, context_length=128):
    from kkoma.config import RunConfig

    cfg = RunConfig()
    cfg.training.micro_batch_size = micro_batch
    cfg.model.context_length = context_length
    return {"config": cfg.to_dict(), "world_size": world_size, "global_step": 10}


def test_resume_rejects_changed_world_size():
    """Blocks shard by ``i % world_size``, so a different world size resumes at a
    different corpus position even though skip_blocks looks right."""

    import pytest

    from kkoma.config import RunConfig
    from scripts._common import check_resume_compatibility

    cfg = RunConfig()
    cfg.training.micro_batch_size = 4
    cfg.model.context_length = 128
    with pytest.raises(ValueError, match="world_size"):
        check_resume_compatibility(_ckpt_payload(world_size=2), cfg, world_size=4)


def test_resume_rejects_changed_batch_geometry():
    import pytest

    from kkoma.config import RunConfig
    from scripts._common import check_resume_compatibility

    cfg = RunConfig()
    cfg.training.micro_batch_size = 8      # checkpoint saved with 4
    cfg.model.context_length = 128
    with pytest.raises(ValueError, match="micro_batch_size"):
        check_resume_compatibility(_ckpt_payload(world_size=2), cfg, world_size=2)


def test_resume_accepts_matching_geometry():
    from kkoma.config import RunConfig
    from scripts._common import check_resume_compatibility

    cfg = RunConfig()
    cfg.training.micro_batch_size = 4
    cfg.model.context_length = 128
    check_resume_compatibility(_ckpt_payload(world_size=2), cfg, world_size=2)


def test_resume_allows_checkpoint_without_config():
    """Older checkpoints carry no config; nothing to compare, so do not block."""

    from kkoma.config import RunConfig
    from scripts._common import check_resume_compatibility

    check_resume_compatibility({"global_step": 5}, RunConfig(), world_size=8)


def test_rng_restores_when_the_payload_was_mapped_to_a_device():
    """`read_checkpoint(map_location=...)` moves every tensor, RNG states too.

    Resuming on GPU therefore handed `set_rng_state` a CUDA tensor and died
    with "RNG state must be a torch.ByteTensor". No shipped run used --resume,
    so it went unnoticed; this pins the coercion back to a CPU ByteTensor.
    """

    import torch

    from kkoma.training.checkpoint import _restore_rng, _rng_states

    states = _rng_states()
    if torch.cuda.is_available():
        moved = dict(states)
        moved["torch"] = states["torch"].to("cuda")
        if states.get("cuda") is not None:
            moved["cuda"] = [s.to("cuda") for s in states["cuda"]]
    else:
        # Same coercion path: a non-uint8 dtype must also be accepted back.
        moved = dict(states, torch=states["torch"].to(torch.int64))

    _restore_rng(moved)          # must not raise
    assert torch.get_rng_state().dtype == torch.uint8
