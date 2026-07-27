"""Weighted mixture over data sources (spec sections 6.5 / 14.1).

Documents from several sources are interleaved according to per-source weights
using a deterministic, seed-based scheme.

``weighting`` decides what the weights are a ratio *of*:

- ``"token"`` (default) — the ratio applies to tokens, which is what the
  configs mean when they say 95/5 or 70/30. Sources whose documents are longer
  would otherwise contribute more tokens per draw, so the sampler divides each
  weight by that source's running mean document length.
- ``"document"`` — the ratio applies to documents, the literal draw
  probability. This is what the shipped 1.3B runs used; keep it to reproduce
  them.

The difference is not cosmetic. Under ``"document"`` the 2026-07 CPT run's
70/30 target was realized as 64/36 in tokens, and the English source drained at
84% of the budget, leaving the last 1,218 steps Korean-only (see
docs/audit-2026-07.md §A-3).
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
        weighting: str = "token",
    ):
        """``shuffle_buffer > 0`` enables seed-based document shuffling inside
        each source (spec section 14.2); the prepare scripts use it while
        writing corpus shards, training reads the fixed shards without it.

        ``weighting`` is ``"token"`` or ``"document"`` — see the module docstring.
        """

        if not sources:
            raise ValueError("mixture requires at least one source")
        if weighting not in ("token", "document"):
            raise ValueError(f"weighting must be 'token' or 'document', got {weighting!r}")
        self.sources = sources
        self.seed = seed
        self.clean_config = clean_config
        self.shuffle_buffer = shuffle_buffer
        self.weighting = weighting
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
        n_docs = 0
        # Running mean document length per source, used only by token weighting.
        # Bytes stand in for tokens: this tokenizer's bytes-per-token is within
        # 0.2% across the two languages (4.594 KO vs 4.587 EN measured on the
        # prepared corpora), so the correction is the same either way and needs
        # no tokenizer here.
        seen = [0] * len(iterators)
        total_len = [0] * len(iterators)

        def draw_weights() -> list[float]:
            if self.weighting == "document":
                return [self.weights[i] for i in active]
            out = []
            for i in active:
                mean_len = (total_len[i] / seen[i]) if seen[i] else 1.0
                out.append(self.weights[i] / max(mean_len, 1e-9))
            # All-zero cannot happen (weights are positive), but guard anyway.
            return out if any(out) else [self.weights[i] for i in active]

        while active:
            idx = rng.choices(active, weights=draw_weights())[0]
            try:
                doc = next(iterators[idx])
            except StopIteration:
                pos = active.index(idx)
                active.pop(pos)
                # Once a source runs dry the remaining probability mass
                # redistributes and the tail of the stream is a different
                # mixture than the head — in the limit, a single source. That
                # shift is invisible in the final totals when the corpora were
                # budgeted to sum to the target, so announce it.
                name = self.sources[idx].name
                remaining = [self.sources[i].name for i in active]
                print(
                    f"[mixture] source {name!r} exhausted after {n_docs:,} documents; "
                    + (
                        f"the rest of the stream comes only from {remaining}"
                        if remaining
                        else "the stream is now empty"
                    ),
                    flush=True,
                )
                continue
            seen[idx] += 1
            total_len[idx] += len(doc)
            n_docs += 1
            yield doc


__all__ = ["MixtureStream"]
