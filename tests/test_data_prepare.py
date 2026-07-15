"""Corpus preparation: seed shuffle (spec 14.2) and holdout split (spec 14.3)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkoma.config import DataSource
from kkoma.data.mixture import MixtureStream
from scripts._prepare import holdout_is_val, prepare_corpus


def _write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for text in docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _make_source(tmp_path, n_docs=500, n_files=4):
    docs = [f"document number {i} with enough characters to pass cleaning" for i in range(n_docs)]
    per_file = n_docs // n_files
    for j in range(n_files):
        _write_jsonl(tmp_path / f"shard_{j:05d}.jsonl", docs[j * per_file : (j + 1) * per_file])
    return DataSource(name="local", path=str(tmp_path / "*.jsonl")), docs


def _read_corpus(out_dir):
    docs = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, name), encoding="utf-8") as f:
            docs.extend(json.loads(line)["text"] for line in f if line.strip())
    return docs


def test_holdout_is_deterministic_and_content_based():
    doc = "the same document always lands on the same side"
    assert all(holdout_is_val(doc, 10) == holdout_is_val(doc, 10) for _ in range(3))
    docs = [f"doc {i} padded to a reasonable document length" for i in range(2000)]
    val_frac = sum(holdout_is_val(d, 10) for d in docs) / len(docs)
    assert 0.05 < val_frac < 0.15  # ~1/10 of documents


def test_shuffle_is_seeded_and_complete(tmp_path):
    source, docs = _make_source(tmp_path)
    order_a = list(MixtureStream([source], seed=1, shuffle_buffer=64))
    order_b = list(MixtureStream([source], seed=1, shuffle_buffer=64))
    order_c = list(MixtureStream([source], seed=2, shuffle_buffer=64))
    unshuffled = list(MixtureStream([source], seed=1))

    assert order_a == order_b  # same seed, same order
    assert order_a != order_c  # different seed, different order
    assert order_a != unshuffled  # shuffling actually reorders
    assert sorted(order_a) == sorted(docs)  # no document lost or duplicated
    assert unshuffled == docs  # shuffle off keeps the fixed on-disk order


def test_train_and_val_corpora_are_disjoint(tmp_path):
    source, docs = _make_source(tmp_path)
    train = prepare_corpus([source], str(tmp_path / "train"), split="train",
                           holdout_mod=10, seed=1, show_progress=False)
    val = prepare_corpus([source], str(tmp_path / "val"), split="val",
                         holdout_mod=10, seed=99, show_progress=False)

    train_docs = set(_read_corpus(tmp_path / "train"))
    val_docs = set(_read_corpus(tmp_path / "val"))
    assert train.documents > 0 and val.documents > 0
    assert not train_docs & val_docs  # disjoint regardless of seed
    assert train_docs | val_docs == set(docs)  # together they cover the input
    assert all(holdout_is_val(d, 10) for d in val_docs)


def test_manifest_records_split_method(tmp_path):
    source, _ = _make_source(tmp_path)
    result = prepare_corpus([source], str(tmp_path / "out"), split="train",
                            holdout_mod=10, shuffle_buffer=32, seed=7, show_progress=False)
    with open(result.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["holdout"]["role"] == "train"
    assert manifest["holdout"]["holdout_mod"] == 10
    assert "sha256" in manifest["holdout"]["method"]
    assert manifest["shuffle"] == {"seed": 7, "buffer_size": 32}
