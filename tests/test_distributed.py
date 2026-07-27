"""DDP and gradient-accumulation equivalence (spec 22.2).

The spec requires these and the suite had neither before the 2026-07 audit, so
the whole distributed path — DDP gradient averaging, the ``no_sync`` window used
during accumulation, and the accumulation arithmetic itself — shipped unverified.

Both properties are the same claim from two directions: splitting a batch, by
accumulation step or by rank, must produce the gradient of the whole batch.
Everything runs on CPU with gloo so it needs no GPU.
"""

from __future__ import annotations

import os

import pytest
import torch

from kkoma.model.model import KkomaModel
from tests.conftest import tiny_config

DIST_AVAILABLE = torch.distributed.is_available() and torch.distributed.is_gloo_available()


def _flat_grad(model) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for _, p in sorted(model.named_parameters())
    ])


def _fixed_batch(cfg, n, seq=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (n, seq), generator=g)
    return ids


def _fresh_model(cfg):
    torch.manual_seed(1234)
    return KkomaModel(cfg)


def test_gradient_accumulation_equals_one_large_batch():
    """accum=N over micro-batches must equal one backward on the full batch.

    This is the arithmetic the trainer relies on to hit a 262,144-token global
    batch on GPUs that cannot hold it: each micro-step scales its loss by
    1/grad_accum before backward.
    """

    cfg = tiny_config()
    ids = _fixed_batch(cfg, 8)

    full = _fresh_model(cfg)
    _, loss = full(ids, labels=ids)
    loss.backward()
    reference = _flat_grad(full)

    accum = _fresh_model(cfg)
    n_micro = 4
    for chunk in ids.chunk(n_micro):
        _, l = accum(chunk, labels=chunk)
        (l / n_micro).backward()
    got = _flat_grad(accum)

    assert torch.allclose(reference, got, atol=1e-5), (
        f"max |diff| = {(reference - got).abs().max().item():.3e}"
    )


def _ddp_worker(rank: int, world_size: int, out_path: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size,
    )
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP

        cfg = tiny_config()
        model = DDP(_fresh_model(cfg))
        ids = _fixed_batch(cfg, 8)
        shard = ids.chunk(world_size)[rank]

        _, loss = model(shard, labels=shard)
        loss.backward()
        if rank == 0:
            torch.save(_flat_grad(model.module), out_path)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not DIST_AVAILABLE, reason="torch.distributed/gloo unavailable")
def test_ddp_averages_gradients_across_ranks(tmp_path):
    """Two ranks on half a batch each must match one process on the whole batch.

    DDP all-reduces gradients with a mean, so per-rank means over equal shards
    compose into the full-batch mean. If that ever stops holding — a changed
    reduction, an unsynced parameter, a bucket that never fires — every
    multi-GPU run silently trains on a different gradient than the single-GPU
    run it is compared against.
    """

    cfg = tiny_config()
    ids = _fixed_batch(cfg, 8)
    single = _fresh_model(cfg)
    _, loss = single(ids, labels=ids)
    loss.backward()
    reference = _flat_grad(single)

    out = str(tmp_path / "rank0_grad.pt")
    torch.multiprocessing.spawn(_ddp_worker, args=(2, out), nprocs=2, join=True)
    got = torch.load(out, weights_only=True)

    assert torch.allclose(reference, got, atol=1e-5), (
        f"max |diff| = {(reference - got).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not DIST_AVAILABLE, reason="torch.distributed/gloo unavailable")
def test_all_reduce_sum_matches_python_sum(tmp_path):
    """The trainer's eval and downstream paths reduce with this exactly once per
    rank; a wrong op would rescale every reported validation loss."""

    out = str(tmp_path / "reduced.pt")
    torch.multiprocessing.spawn(_reduce_worker, args=(2, out), nprocs=2, join=True)
    assert torch.equal(torch.load(out, weights_only=True), torch.tensor([3.0, 30.0]))


def _reduce_worker(rank: int, world_size: int, out_path: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from kkoma.training.distributed import all_reduce_sum

        t = torch.tensor([1.0, 10.0]) if rank == 0 else torch.tensor([2.0, 20.0])
        r = all_reduce_sum(t)
        if rank == 0:
            torch.save(r, out_path)
    finally:
        torch.distributed.destroy_process_group()
