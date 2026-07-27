"""Corpus preparation: seed shuffle (spec 14.2) and holdout split (spec 14.3)."""

from __future__ import annotations

import json
import os
import sys

import pytest

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


def _uniform_length_source(tmp_path, name, char_len, n_docs):
    path = tmp_path / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_docs):
            f.write(json.dumps({"text": "가" * char_len}, ensure_ascii=False) + "\n")
    return DataSource(name=name, path=str(path), weight=0.5)


def _realized_ratio(stream, take):
    by_len = {}
    for i, doc in enumerate(stream):
        if i >= take:
            break
        by_len[len(doc)] = by_len.get(len(doc), 0) + len(doc)
    total = sum(by_len.values())
    return {k: v / total for k, v in by_len.items()}


def test_document_weighting_misses_the_token_target(tmp_path):
    """The pre-2026-07 behaviour, kept for reproducing the shipped runs.

    Equal per-document weights over sources with 1:4 document lengths give a 1:4
    token split, not 1:1 — which is how the 1.3B CPT run's 70/30 target was
    realized as 64/36 (docs/audit-2026-07.md §A-3).
    """

    srcs = [_uniform_length_source(tmp_path, "short", 100, 4000),
            _uniform_length_source(tmp_path, "long", 400, 4000)]
    ratio = _realized_ratio(MixtureStream(srcs, seed=42, weighting="document"), 4000)
    assert ratio[100] == pytest.approx(0.2, abs=0.03)
    assert ratio[400] == pytest.approx(0.8, abs=0.03)


def test_token_weighting_hits_the_token_target(tmp_path):
    """Default weighting makes the configured ratio mean tokens, as documented."""

    srcs = [_uniform_length_source(tmp_path, "short", 100, 4000),
            _uniform_length_source(tmp_path, "long", 400, 4000)]
    ratio = _realized_ratio(MixtureStream(srcs, seed=42, weighting="token"), 4000)
    assert ratio[100] == pytest.approx(0.5, abs=0.03)
    assert ratio[400] == pytest.approx(0.5, abs=0.03)


def test_missing_glob_raises_instead_of_yielding_nothing(tmp_path):
    """A path that matches no files is a config error, not an empty corpus.

    Silently returning an empty stream sent training straight to "data stream
    exhausted" and validation to NaN, both far from the cause.
    """

    src = DataSource(name="gone", path=str(tmp_path / "does_not_exist" / "*.jsonl"))
    with pytest.raises(FileNotFoundError, match="no files matched"):
        list(MixtureStream([src], seed=1))


# ---------------------------------------------------------------------------
# Downstream task formatters (scripts/prepare_downstream_data.py). Each maps a
# raw dataset row to the frozen record shape; the choice delimiter is a leading
# space and the gold index is 0-based.
# ---------------------------------------------------------------------------

from scripts.prepare_downstream_data import (  # noqa: E402
    _boolq,
    _kobest_boolq,
    _kobest_copa,
    _kobest_sentineg,
    _kobest_wic,
    _openbookqa,
    _piqa,
    _winogrande,
)


def test_piqa_formats_two_choice_record():
    rec = _piqa({"goal": "How to open a jar?", "sol1": "twist lid", "sol2": "smash it", "label": 1}, 0)
    assert rec["context"] == "Question: How to open a jar?\nAnswer:"
    assert rec["choices"] == [" twist lid", " smash it"]
    assert rec["label"] == 1


def test_piqa_drops_unlabeled_test_rows():
    assert _piqa({"goal": "g", "sol1": "a", "sol2": "b", "label": -1}, 0) is None


def test_boolq_maps_label_to_no_yes():
    rec = _boolq({"passage": "Cats are mammals.", "question": "is a cat a mammal", "label": 1}, 3)
    assert rec["choices"] == [" no", " yes"]  # index 1 == yes == gold
    assert rec["context"].endswith("Answer:")
    assert rec["label"] == 1


def test_winogrande_partial_uses_per_option_context():
    row = {
        "sentence": "The trophy did not fit in the case because _ was too big.",
        "option1": "the trophy",
        "option2": "the case",
        "answer": "1",
    }
    rec = _winogrande(row, 0)
    prefix = "The trophy did not fit in the case because "
    assert rec["contexts"] == [prefix + "the trophy", prefix + "the case"]
    # Shared continuation (the suffix after the blank), identical for both options.
    assert rec["choices"] == [" was too big.", " was too big."]
    assert rec["label"] == 0  # answer "1" -> index 0


def test_winogrande_drops_empty_answer():
    row = {"sentence": "A _ B.", "option1": "x", "option2": "y", "answer": ""}
    assert _winogrande(row, 0) is None


def test_openbookqa_indexes_answer_key():
    row = {
        "id": "q1",
        "question_stem": "The sun is a",
        "choices": {"text": ["planet", "star", "moon", "comet"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    }
    rec = _openbookqa(row, 0)
    assert rec["context"] == "The sun is a"
    assert rec["choices"] == [" planet", " star", " moon", " comet"]
    assert rec["label"] == 1


def test_kobest_copa_selects_connector():
    cause = _kobest_copa(
        {"premise": "그는 넘어졌다.", "question": "원인", "alternative_1": "돌", "alternative_2": "물", "label": 0}, 0
    )
    assert "왜냐하면" in cause["context"]
    effect = _kobest_copa(
        {"premise": "그는 넘어졌다.", "question": "결과", "alternative_1": "돌", "alternative_2": "물", "label": 1}, 0
    )
    assert "그래서" in effect["context"]
    assert cause["choices"] == [" 돌", " 물"]


def test_kobest_boolq_and_sentineg_and_wic_choices():
    b = _kobest_boolq({"paragraph": "고양이는 포유류다.", "question": "고양이는 동물인가?", "label": 1}, 0)
    assert b["choices"] == [" 아니오", " 예"] and b["label"] == 1
    s = _kobest_sentineg({"sentence": "정말 최고의 영화였다.", "label": 1}, 0)
    assert s["choices"] == [" 부정", " 긍정"] and s["context"].startswith("문장:")
    w = _kobest_wic(
        {"word": "배", "context_1": "사과와 배를 먹었다.", "context_2": "배가 아프다.", "label": 0}, 0
    )
    assert w["choices"] == [" 아니오", " 예"]
    assert "배가 같은 뜻으로 쓰였나?" in w["context"]


def test_parquet_is_detected_by_file_extension_not_glob_spelling(tmp_path):
    """Dispatch must follow the matched files, not how the pattern was written.

    A glob that does not end in a recognized suffix used to fall through to the
    JSONL reader, which swallows every parse error — real parquet shards then
    produced an empty stream with nothing to explain it.
    """

    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kkoma.data.streaming import stream_source

    d = tmp_path / "shards"
    d.mkdir()
    docs = [f"parquet document number {i} with enough characters" for i in range(6)]
    pq.write_table(pa.table({"text": docs}), d / "part-0.parquet")

    # Pattern ends in "*", not ".parquet" — the old string test missed this.
    src = DataSource(name="pq", path=str(d / "part-*"))
    assert sorted(stream_source(src)) == sorted(docs)
