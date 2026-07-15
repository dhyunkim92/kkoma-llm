"""Weighted mixture over data sources (spec sections 6.5 / 14.1).

Documents from several sources are interleaved according to per-source
weights using a deterministic, seed-based scheme. Weights are interpreted as
sampling probabilities; the realized token ratio is reported by the manifest
builder so it can be verified against the target (e.g. 95/5, 70/30).
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from kkoma.config import DataSource
from kkoma.data.preprocessing import CleanConfig
from kkoma.data.streaming import stream_source


class MixtureStream:
    """Deterministically interleave multiple document streams by weight."""

    def __init__(
        self,
        sources: list[DataSource],
        seed: int = 42,
        clean_config: Optional[CleanConfig] = None,
        shuffle_buffer: int = 0,
    ):
        """``shuffle_buffer > 0`` enables seed-based document shuffling inside
        each source (spec section 14.2); the prepare scripts use it while
        writing corpus shards, training reads the fixed shards without it."""

        if not sources:
            raise ValueError("mixture requires at least one source")
        self.sources = sources
        self.seed = seed
        self.clean_config = clean_config
        self.shuffle_buffer = shuffle_buffer
        total = sum(s.weight for s in sources)
        self.weights = [s.weight / total for s in sources]

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        iterators = [
            stream_source(
                s,
                self.clean_config,
                shuffle_seed=(self.seed + i) if self.shuffle_buffer > 0 else None,
                shuffle_buffer=self.shuffle_buffer,
            )
            for i, s in enumerate(self.sources)
        ]
        active = list(range(len(iterators)))
        weights = list(self.weights)
        while active:
            idx = rng.choices(active, weights=[weights[i] for i in active])[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                pos = active.index(idx)
                active.pop(pos)


__all__ = ["MixtureStream"]
