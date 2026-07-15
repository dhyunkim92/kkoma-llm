"""Integration tests (spec 22.2 / 22.3): overfit, packing, scheduler shape."""

from __future__ import annotations

import torch

from kkoma.config import OptimizerConfig, RunConfig, SchedulerConfig
from kkoma.data.packing import PackedDataset, collate_blocks, pack_token_stream
from kkoma.model.model import KkomaModel
from kkoma.training.distributed import DistInfo
from kkoma.training.optimizer import build_optimizer
from kkoma.training.scheduler import build_scheduler
from kkoma.training.trainer import Logger, Trainer
from tests.conftest import tiny_config


def test_overfit_small_batch():
    """A tiny model must drive loss near zero on a single repeated batch."""

    torch.manual_seed(0)
    model = KkomaModel(tiny_config())
    opt = build_optimizer(model, OptimizerConfig(learning_rate=1e-3))
    batch = torch.randint(0, model.config.vocab_size, (1, 16))
    model.train()
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(batch, labels=batch)
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_pack_token_stream_block_size():
    stream = range(25)
    blocks = list(pack_token_stream(stream, block_size=8))
    assert len(blocks) == 3  # 25 // 8, remainder dropped
    assert all(len(b) == 8 for b in blocks)


class _WordTokenizer:
    """Whitespace-token stand-in exposing the encode_document interface."""

    def encode_document(self, doc):
        return [int(t) for t in doc.split()]


def _blocks(dataset):
    return [b["input_ids"].tolist() for b in dataset]


def test_packed_dataset_skip_blocks_resumes_exactly():
    """Fast-forward (spec 18.3): skipping k blocks lands on the exact next batch."""

    docs = [" ".join(str(i) for i in range(d * 10, d * 10 + 10)) for d in range(20)]
    make = lambda skip: PackedDataset(docs, _WordTokenizer(), block_size=8, skip_blocks=skip)

    full = _blocks(make(0))
    assert len(full) > 5
    for k in (1, 3, 5):
        assert _blocks(make(k)) == full[k:]


def test_packed_dataset_skip_counts_own_shard_only():
    docs = [" ".join(str(i) for i in range(d * 10, d * 10 + 10)) for d in range(20)]
    tok = _WordTokenizer()
    shard = PackedDataset(docs, tok, block_size=8, rank=1, world_size=2)
    skipped = PackedDataset(docs, tok, block_size=8, rank=1, world_size=2, skip_blocks=2)
    assert _blocks(skipped) == _blocks(shard)[2:]


def test_collate_blocks_shapes():
    batch = [
        {"input_ids": torch.arange(8), "labels": torch.arange(8)} for _ in range(4)
    ]
    out = collate_blocks(batch)
    assert out["input_ids"].shape == (4, 8)
    assert out["labels"].shape == (4, 8)


def _make_trainer(grad_accum=2):
    torch.manual_seed(0)
    cfg = RunConfig()
    cfg.model = tiny_config()
    cfg.training.micro_batch_size = 1
    cfg.training.grad_accum_steps = grad_accum
    cfg.training.precision = "fp32"
    cfg.logging.backend = "none"
    model = KkomaModel(cfg.model)
    opt = build_optimizer(model, cfg.optimizer)
    sch = build_scheduler(opt, cfg.scheduler, total_steps=10)
    dist = DistInfo()
    return Trainer(
        config=cfg, model=model, raw_model=model, optimizer=opt, scheduler=sch,
        train_loader=[], dist=dist, device=torch.device("cpu"),
        val_loaders={}, logger=Logger(cfg, dist),
    )


def _batch(cfg, seq=16):
    ids = torch.randint(0, cfg.model.vocab_size, (1, seq))
    return {"input_ids": ids, "labels": ids.clone()}


def test_train_step_discards_partial_accumulation_window():
    """Exhaustion mid-window returns None (all ranks stop together under DDP)."""

    trainer = _make_trainer(grad_accum=2)
    out = trainer._train_step(iter([_batch(trainer.config)]))  # 1 of 2 batches
    assert out is None
    assert trainer.state.global_step == 0


def test_train_step_full_window_steps():
    trainer = _make_trainer(grad_accum=2)
    batches = [_batch(trainer.config), _batch(trainer.config)]
    out = trainer._train_step(iter(batches))
    assert out is not None and out["skipped"] == 0
    assert trainer.state.global_step == 1


def test_train_step_skips_on_non_finite_loss():
    trainer = _make_trainer(grad_accum=1)
    weight = trainer.raw_model.token_embedding.weight
    trainer._forward_loss = lambda batch: weight.sum() * float("nan")
    out = trainer._train_step(iter([_batch(trainer.config)]))
    assert out is not None and out["skipped"] == 1
    assert trainer.state.global_step == 0
    assert trainer.state.skipped_steps == 1


def test_scheduler_warmup_then_decay():
    model = KkomaModel(tiny_config())
    opt = build_optimizer(model, OptimizerConfig(learning_rate=1.0))
    sched = build_scheduler(opt, SchedulerConfig(warmup_ratio=0.1, min_lr_ratio=0.1), total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[10]              # warming up
    assert lrs[10] >= max(lrs) - 1e-9    # peak around end of warmup
    assert lrs[-1] < lrs[10]             # decayed
    assert lrs[-1] >= 0.1 - 1e-6         # floored at min_lr_ratio
