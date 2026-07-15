"""Distributed Data Parallel helpers (spec section 16).

One process per GPU, NCCL backend. Reads the standard ``torchrun`` environment
variables. All helpers degrade gracefully to single-process when distributed is
not initialized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = "nccl") -> DistInfo:
    """Initialize the process group if launched under torchrun."""

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
        return DistInfo(rank=0, local_rank=local_rank, world_size=1, distributed=False)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    elif backend == "nccl":
        backend = "gloo"  # NCCL needs GPUs; torchrun on a CPU host falls back
    # Explicit timeout so a desynchronized collective fails loudly instead of
    # hanging forever on the NCCL default.
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    return DistInfo(
        rank=rank, local_rank=local_rank, world_size=world_size, distributed=True
    )


def cleanup_distributed(barrier: bool = True) -> None:
    """Tear down the process group.

    Pass ``barrier=False`` on the exception path: the failed rank would wait
    at the barrier for peers that are stuck inside a collective it never
    joined, turning one rank's crash into a full-job hang.
    """

    if dist.is_available() and dist.is_initialized():
        if barrier:
            dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


__all__ = [
    "DistInfo",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "all_reduce_mean",
    "all_reduce_sum",
]
