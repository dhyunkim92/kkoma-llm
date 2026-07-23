"""Tiny shared console helpers for a consistent CLI look across scripts.

Staged / one-shot scripts (evaluate, sample, prepare_downstream, leaderboard,
train_tokenizer) all use the same shape so their output reads the same way:

    ▸ <section> …
        <result line>
        (<elapsed>s)
    ✓ <final message>

Long streaming/looping scripts (training, corpus preparation) use a live tqdm
progress bar instead — that is the right tool for millions of steps/documents,
and those already carry it.

Stdlib-only on purpose: scripts/leaderboard.py imports this without pulling in
torch, so the helper must stay dependency-free.
"""

from __future__ import annotations

import time


def section(title: str) -> float:
    """Print a section header and return a start time to hand to ``done()``."""
    print(f"\n▸ {title} …", flush=True)
    return time.perf_counter()


def line(msg: str) -> None:
    """Print an indented result line under the current section."""
    print(f"    {msg}", flush=True)


def done(t0: float) -> None:
    """Print the elapsed time for a section started by ``section()``."""
    print(f"    ({time.perf_counter() - t0:.1f}s)", flush=True)


def ok(msg: str) -> None:
    """Print the final success line."""
    print(f"\n✓ {msg}", flush=True)


__all__ = ["section", "line", "done", "ok"]
