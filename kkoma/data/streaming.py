"""Document streaming from local JSONL/Parquet shards or HF ``datasets``.

Each source yields cleaned text strings. Streaming keeps memory bounded for the
16B-token master corpus.

Shuffling (spec section 14.2, "corpus 생성 전에 문서 순서를 seed 기반으로 섞고
고정한다") is seed-based and applied only when a shuffle buffer is requested:
the prepare_* scripts shuffle while writing the corpus shards, and training
then reads those shards in their fixed order so runs stay reproducible and
resumable. For HF streaming sources this uses ``datasets``' own shuffle (shard
order + reservoir buffer); for local shards the file order is shuffled and a
buffer shuffle is applied over documents.
"""

from __future__ import annotations

import glob
import json
import random
from typing import Iterator, Optional

from kkoma.config import DataSource
from kkoma.data.preprocessing import CleanConfig, clean_document


def _buffer_shuffle(docs: Iterator[str], seed: int, buffer_size: int) -> Iterator[str]:
    """Approximate shuffle over a stream using a fixed-size buffer.

    Same scheme as ``datasets``' streaming shuffle: keep ``buffer_size`` docs,
    yield a random one and replace it with the next from the stream.
    """

    rng = random.Random(seed)
    buffer: list[str] = []
    for doc in docs:
        if len(buffer) < buffer_size:
            buffer.append(doc)
            continue
        idx = rng.randrange(buffer_size)
        yield buffer[idx]
        buffer[idx] = doc
    rng.shuffle(buffer)
    yield from buffer


def _shard_files(path: str, seed: Optional[int]) -> list[str]:
    files = sorted(glob.glob(path))
    if seed is not None:
        random.Random(seed).shuffle(files)
    return files


def _iter_jsonl(path: str, text_key: str, seed: Optional[int] = None) -> Iterator[str]:
    for file in _shard_files(path, seed):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_key)
                if text:
                    yield text


def _iter_parquet(path: str, text_key: str, seed: Optional[int] = None) -> Iterator[str]:
    import pyarrow.parquet as pq

    for file in _shard_files(path, seed):
        table = pq.read_table(file, columns=[text_key])
        for value in table.column(text_key).to_pylist():
            if value:
                yield value


def _iter_hf(source: DataSource, seed: Optional[int] = None, buffer_size: int = 0) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        source.name,
        source.subset,
        split=source.split,
        revision=source.revision,
        streaming=True,
    )
    if seed is not None and buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for example in ds:
        text = example.get(source.text_key)
        if text:
            yield text


def stream_source(
    source: DataSource,
    clean_config: Optional[CleanConfig] = None,
    *,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer: int = 0,
) -> Iterator[str]:
    """Yield cleaned documents from a single source.

    With ``shuffle_seed`` set and ``shuffle_buffer > 0`` the document order is
    seed-shuffled; otherwise the source is read in its natural, fixed order.
    """

    shuffle = shuffle_seed is not None and shuffle_buffer > 0
    seed = shuffle_seed if shuffle else None

    if source.path and source.path.endswith((".parquet", ".parquet/*", "*.parquet")):
        raw = _iter_parquet(source.path, source.text_key, seed)
    elif source.path:
        raw = _iter_jsonl(source.path, source.text_key, seed)
    else:
        raw = _iter_hf(source, seed, shuffle_buffer)

    if shuffle and source.path:
        # HF streaming shuffles internally; local shards get the buffer here.
        raw = _buffer_shuffle(raw, seed + 1, shuffle_buffer)

    for text in raw:
        cleaned = clean_document(text, clean_config)
        if cleaned is not None:
            yield cleaned


__all__ = ["stream_source"]
