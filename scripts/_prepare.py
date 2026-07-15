"""Shared corpus-preparation helpers.

Streams a weighted mixture of sources, applies minimal cleaning, samples up to a
byte or token budget, writes JSONL shards, and emits a reproducible data
manifest (spec section 25). Used by every ``prepare_*`` script.

Train/validation separation (spec section 14.3) is a document-level hash
holdout: each document is assigned to the validation pool iff
``sha256(text) % holdout_mod == 0``. The assignment depends only on the
document content, so train (``split="train"``) and validation (``split="val"``)
corpora are disjoint regardless of stream order, seed, or when each corpus is
prepared, and exact-duplicate documents always land on the same side.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from kkoma.config import DataSource
from kkoma.data.mixture import MixtureStream
from kkoma.data.preprocessing import CleanConfig
from kkoma.tokenizer.utils import KkomaTokenizer


@dataclass
class PrepareResult:
    documents: int
    bytes: int
    tokens: int
    shards: list[str]
    manifest_path: str


def _shard_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"shard_{idx:05d}.jsonl")


def holdout_is_val(text: str, holdout_mod: int) -> bool:
    """Deterministic document-level validation assignment (spec section 14.3)."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % holdout_mod == 0


def hard_exit(code: int = 0) -> None:
    """Flush stdio and exit immediately, skipping interpreter finalization.

    Hugging Face ``datasets`` streaming keeps background HTTP prefetch threads
    alive. When we stop early (token/byte budget reached) those threads can race
    with interpreter shutdown and crash the process with
    ``Fatal Python error: PyGILState_Release ... no thread-state``. All shards
    and manifests are already written and flushed by this point, so a hard exit
    is safe and avoids the shutdown race entirely.
    """

    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---------------------------------------------------------------------------
# Progress display (overall budget bar + per-shard document bar)
# ---------------------------------------------------------------------------


class _NullProgress:
    """No-op progress used when display is disabled or tqdm is unavailable."""

    def advance(self, amount: int) -> None: ...
    def new_shard(self, idx: int, n_shards: int) -> None: ...
    def set_shards(self, n_shards: int) -> None: ...
    def close(self) -> None: ...


class _TqdmProgress:
    def __init__(self, tqdm, total, unit, docs_per_shard):
        self.docs_per_shard = docs_per_shard
        self.overall = tqdm(
            total=total,
            unit=unit,
            unit_scale=True,
            desc="overall",
            position=0,
            dynamic_ncols=True,
        )
        self.shard = tqdm(
            total=docs_per_shard,
            unit="doc",
            desc="shard 00000",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

    def advance(self, amount: int) -> None:
        self.overall.update(amount)
        self.shard.update(1)

    def new_shard(self, idx: int, n_shards: int) -> None:
        self.shard.reset(total=self.docs_per_shard)
        self.shard.set_description(f"shard {idx:05d}")
        self.set_shards(n_shards)

    def set_shards(self, n_shards: int) -> None:
        self.overall.set_postfix(shards=n_shards, refresh=False)

    def close(self) -> None:
        self.shard.close()
        self.overall.close()


def _make_progress(show: bool, total, unit: str, docs_per_shard: int):
    if not show:
        return _NullProgress()
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullProgress()
    return _TqdmProgress(tqdm, total, unit, docs_per_shard)


def prepare_corpus(
    sources: list[DataSource],
    out_dir: str,
    *,
    max_bytes: Optional[int] = None,
    max_tokens: Optional[int] = None,
    tokenizer: Optional[KkomaTokenizer] = None,
    seed: int = 42,
    docs_per_shard: int = 100_000,
    min_chars: int = 16,
    show_progress: bool = True,
    split: str = "all",
    holdout_mod: int = 100,
    shuffle_buffer: int = 10_000,
) -> PrepareResult:
    """Write a sampled, cleaned corpus to JSONL shards and a manifest.

    ``split`` selects which side of the document-level hash holdout is kept:
    ``"train"`` drops validation-pool documents, ``"val"`` keeps only them, and
    ``"all"`` keeps everything (tokenizer data, where no holdout is needed).
    ``shuffle_buffer > 0`` seed-shuffles the document order before writing
    (spec section 14.2), so the resulting shards are a fixed shuffled corpus.

    Two progress bars are shown (when ``show_progress`` and tqdm are available):
    an overall bar toward the byte/token budget (with ETA and a running shard
    count) and a per-shard bar over the documents written to the current shard.
    """

    if max_tokens is not None and tokenizer is None:
        raise ValueError("max_tokens requires a tokenizer for counting")
    if split not in ("all", "train", "val"):
        raise ValueError(f"split must be 'all', 'train' or 'val', got {split!r}")
    os.makedirs(out_dir, exist_ok=True)

    clean = CleanConfig(min_chars=min_chars)
    stream = MixtureStream(
        sources, seed=seed, clean_config=clean, shuffle_buffer=shuffle_buffer
    )

    # The overall bar tracks tokens when a token budget is set, else bytes.
    use_tokens = max_tokens is not None
    progress = _make_progress(
        show_progress,
        total=max_tokens if use_tokens else max_bytes,
        unit="tok" if use_tokens else "B",
        docs_per_shard=docs_per_shard,
    )

    n_docs = n_bytes = n_tokens = 0
    shard_idx = 0
    shards: list[str] = []
    fh = open(_shard_path(out_dir, shard_idx), "w", encoding="utf-8")
    shards.append(_shard_path(out_dir, shard_idx))
    progress.set_shards(len(shards))

    try:
        for text in stream:
            if split != "all" and holdout_is_val(text, holdout_mod) != (split == "val"):
                continue
            doc_bytes = len(text.encode("utf-8"))
            doc_tokens = len(tokenizer.encode(text)) if tokenizer is not None else 0

            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_bytes += doc_bytes
            n_tokens += doc_tokens
            progress.advance(doc_tokens if use_tokens else doc_bytes)

            if n_docs % docs_per_shard == 0:
                fh.close()
                shard_idx += 1
                path = _shard_path(out_dir, shard_idx)
                fh = open(path, "w", encoding="utf-8")
                shards.append(path)
                progress.new_shard(shard_idx, len(shards))

            if max_bytes is not None and n_bytes >= max_bytes:
                break
            if max_tokens is not None and n_tokens >= max_tokens:
                break
    finally:
        fh.close()
        progress.close()

    manifest_path = _write_manifest(
        out_dir, sources, seed, n_docs, n_bytes, n_tokens, shards,
        split=split, holdout_mod=holdout_mod, shuffle_buffer=shuffle_buffer,
    )
    return PrepareResult(n_docs, n_bytes, n_tokens, shards, manifest_path)


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    out_dir, sources, seed, n_docs, n_bytes, n_tokens, shards,
    *, split="all", holdout_mod=100, shuffle_buffer=0,
) -> str:
    manifest = {
        "sources": [
            {
                "name": s.name,
                "revision": s.revision,
                "subset": s.subset,
                "weight": s.weight,
                "split": s.split,
            }
            for s in sources
        ],
        "sampling_seed": seed,
        "shuffle": {"seed": seed, "buffer_size": shuffle_buffer},
        "holdout": {
            "role": split,
            "method": "document-level: sha256(text)[:8] % holdout_mod == 0 -> val",
            "holdout_mod": holdout_mod,
        },
        "document_count": n_docs,
        "byte_count": n_bytes,
        "token_count": n_tokens,
        "shards": [
            {"path": p, "sha256": _checksum(p)} for p in shards if os.path.exists(p)
        ],
    }
    path = os.path.join(out_dir, "data_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


__all__ = ["prepare_corpus", "PrepareResult", "holdout_is_val", "hard_exit"]
