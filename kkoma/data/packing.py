"""Token packing into fixed-length training blocks.

Documents are tokenized as ``<|bos|> ... <|eos|>`` and concatenated, then cut
into blocks of ``context_length`` tokens. Labels equal the inputs; the model's
forward shifts internally for next-token prediction. Padding is avoided by
construction (a small remainder at the end of the stream is dropped).

``PackedDataset`` is a ``torch.utils.data.IterableDataset`` and shards across
DDP ranks and DataLoader workers so each block is produced exactly once.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from kkoma.tokenizer.utils import KkomaTokenizer


def pack_token_stream(
    token_iter: Iterable[int], block_size: int
) -> Iterator[list[int]]:
    """Yield consecutive blocks of ``block_size`` token ids from a flat stream."""

    buffer: list[int] = []
    for tid in token_iter:
        buffer.append(tid)
        if len(buffer) >= block_size:
            yield buffer[:block_size]
            buffer = buffer[block_size:]


class PackedDataset(IterableDataset):
    def __init__(
        self,
        documents: Iterable[str],
        tokenizer: KkomaTokenizer,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
        skip_blocks: int = 0,
    ):
        """``skip_blocks`` drops that many of this shard's blocks before
        yielding, used to fast-forward a resumed run to the exact next batch
        (spec 18.3). The stream is deterministic, so skipping the consumed
        prefix reproduces the un-interrupted data order; the skipped text is
        still tokenized once, which is the cost of the fast-forward."""

        self.documents = documents
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.skip_blocks = skip_blocks

    def _token_stream(self) -> Iterator[int]:
        for doc in self.documents:
            for tid in self.tokenizer.encode_document(doc):
                yield tid

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        n_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        # Combine DDP rank and DataLoader worker into one shard index space so
        # every block is emitted by exactly one (rank, worker) pair.
        total_shards = self.world_size * n_workers
        shard_id = self.rank * n_workers + worker_id

        skipped = 0
        for i, block in enumerate(pack_token_stream(self._token_stream(), self.block_size)):
            if i % total_shards == shard_id:
                if skipped < self.skip_blocks:
                    skipped += 1
                    continue
                ids = torch.tensor(block, dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


def collate_blocks(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


__all__ = ["PackedDataset", "pack_token_stream", "collate_blocks"]
