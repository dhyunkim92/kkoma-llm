"""Fixed downstream evaluation sets (spec 21.3).

Samples a small, frozen set of multiple-choice questions per task and writes it
to JSONL. Sampling happens exactly once, here: training reads these files as-is,
so every evaluation across every step and every model size scores the identical
questions. That is what makes the curves comparable.

Each task is capped at 500 questions. At that size the standard error on an
accuracy near chance is about 2%, which is coarse, so the numbers are for
watching a trend rather than for quoting. Downstream metrics never pick the best
checkpoint; validation loss does (spec 21.3: focus on movement across model size
and stage, not the absolute score).

Records are written pre-formatted, prompt and all:

    {"id": ..., "context": "...", "choices": [" ...", ...], "label": 0}

The leading space on each choice is the continuation delimiter and is part of
the frozen format, so the scorer stays dumb: it concatenates context + choice
and scores, with no per-task logic. Formats follow lm-evaluation-harness so the
English numbers line up with published ones.

After a task is written its source download is deleted (``--keep-source`` opts
out). Only the task repos in the registry below are touched; other caches,
FineWeb included, are left alone.

Usage:
    python scripts/prepare_downstream_data.py --output-dir data/downstream
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from typing import Callable, Optional

from scripts._console import done, line, ok, section
from scripts._prepare import hard_exit


# ---------------------------------------------------------------------------
# Prompt formatting (one function per task; all return the frozen record shape)
# ---------------------------------------------------------------------------


def _hellaswag_clean(text: str) -> str:
    """The lm-evaluation-harness HellaSwag text normalizer.

    The corpus carries ActivityNet/WikiHow markup ("[title]", "[substeps]")
    that would otherwise be scored as if it were prose.
    """

    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _hellaswag(row: dict, idx: int) -> Optional[dict]:
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    return {
        "id": str(row["ind"]),
        # `label` ships as a string ('3'), not an int.
        "label": int(row["label"]),
        "context": _hellaswag_clean(row["activity_label"] + ": " + ctx),
        "choices": [" " + _hellaswag_clean(e) for e in row["endings"]],
    }


def _arc_easy(row: dict, idx: int) -> Optional[dict]:
    # answerKey mixes numbering schemes across the split ('A'..'E' and '1'..'4'),
    # so the gold index has to come from choices.label rather than from ord().
    labels = list(row["choices"]["label"])
    key = row["answerKey"]
    if key not in labels:
        return None  # no usable gold answer; dropped and reported
    return {
        "id": str(row["id"]),
        "label": labels.index(key),
        # Choice count varies (4 for 2,365 of 2,376, but also 3 and 5). The
        # originals are kept as they are, so the scorer must accept any count.
        "context": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + t.strip() for t in row["choices"]["text"]],
    }


def _kobest_hellaswag(row: dict, idx: int) -> Optional[dict]:
    return {
        "id": str(idx),
        "label": int(row["label"]),
        "context": row["context"].strip(),
        "choices": [" " + row[f"ending_{i}"].strip() for i in range(1, 5)],
    }


def _piqa(row: dict, idx: int) -> Optional[dict]:
    # test ships label -1 (leaderboard-only); only validation has usable gold.
    if int(row["label"]) not in (0, 1):
        return None
    return {
        "id": str(idx),
        "label": int(row["label"]),
        "context": f"Question: {row['goal']}\nAnswer:",
        "choices": [" " + row["sol1"].strip(), " " + row["sol2"].strip()],
    }


def _boolq(row: dict, idx: int) -> Optional[dict]:
    # SuperGLUE BoolQ: yes/no reading comprehension. test ships label -1.
    if int(row["label"]) not in (0, 1):
        return None
    return {
        "id": str(row.get("idx", idx)),
        "label": int(row["label"]),  # 0 -> no, 1 -> yes
        "context": f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:",
        "choices": [" no", " yes"],
    }


def _winogrande(row: dict, idx: int) -> Optional[dict]:
    # Partial scoring (lm-eval-harness): each option fills the sentence's "_"
    # blank to form a per-option *context*, and the shared suffix after the
    # blank is the single continuation scored under each. answer "1"/"2" is the
    # 1-based gold option; the empty-answer test split is dropped.
    answer = str(row["answer"])
    if answer not in ("1", "2"):
        return None
    sentence = row["sentence"]
    if "_" not in sentence:
        return None
    cut = sentence.index("_")
    prefix, suffix = sentence[:cut], sentence[cut + 1 :]
    if not suffix.strip():
        return None  # blank at the very end: no continuation to score
    continuation = " " + suffix.strip()
    return {
        "id": str(idx),
        "label": int(answer) - 1,
        "contexts": [prefix + row["option1"], prefix + row["option2"]],
        "choices": [continuation, continuation],
    }


def _openbookqa(row: dict, idx: int) -> Optional[dict]:
    labels = list(row["choices"]["label"])
    key = row["answerKey"].strip()
    if key not in labels:
        return None
    return {
        "id": str(row.get("id", idx)),
        "label": labels.index(key),
        "context": row["question_stem"].strip(),
        "choices": [" " + t.strip() for t in row["choices"]["text"]],
    }


_KOBEST_COPA_CONNECTOR = {"원인": "왜냐하면", "결과": "그래서"}


def _kobest_copa(row: dict, idx: int) -> Optional[dict]:
    # question "원인"(cause)/"결과"(effect) picks the Korean connector; the two
    # alternatives are the choices (label already 0-based).
    connector = _KOBEST_COPA_CONNECTOR.get(row["question"].strip(), "")
    return {
        "id": str(idx),
        "label": int(row["label"]),
        "context": f"{row['premise']} {connector}".rstrip(),
        "choices": [" " + row["alternative_1"].strip(), " " + row["alternative_2"].strip()],
    }


def _kobest_boolq(row: dict, idx: int) -> Optional[dict]:
    return {
        "id": str(idx),
        "label": int(row["label"]),  # 0 -> 아니오, 1 -> 예
        "context": f"{row['paragraph']} 질문: {row['question']} 답변:",
        "choices": [" 아니오", " 예"],
    }


def _kobest_sentineg(row: dict, idx: int) -> Optional[dict]:
    return {
        "id": str(idx),
        "label": int(row["label"]),  # 0 -> 부정, 1 -> 긍정
        "context": f"문장: {row['sentence']} 긍부정:",
        "choices": [" 부정", " 긍정"],
    }


def _kobest_wic(row: dict, idx: int) -> Optional[dict]:
    return {
        "id": str(idx),
        "label": int(row["label"]),  # 0 -> 아니오(다른 뜻), 1 -> 예(같은 뜻)
        "context": (
            f"문장1: {row['context_1']} 문장2: {row['context_2']} "
            f"두 문장에서 {row['word']}가 같은 뜻으로 쓰였나?"
        ),
        "choices": [" 아니오", " 예"],
    }


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    name: str
    repo: str
    subset: Optional[str]
    split: str
    language: str  # drives the en_avg / ko_avg aggregates during training
    to_record: Callable[[dict, int], Optional[dict]]
    n_sample: Optional[int]  # None keeps the split whole
    stratified: bool  # equal count per gold label
    prompt_format: str  # recorded in the manifest
    # Pinned so re-running this script rebuilds the identical questions. The
    # sets live under data/, which is gitignored, so the pin is the only thing
    # standing between an upstream edit and a silently different benchmark
    # halfway through a model family.
    revision: str


TASKS: list[TaskSpec] = [
    TaskSpec(
        name="hellaswag",
        revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
        repo="Rowan/hellaswag",
        subset=None,
        # The `test` split ships with empty labels (it is leaderboard-only), so
        # `validation` is the only scorable split and is what published
        # HellaSwag numbers refer to.
        split="validation",
        language="en",
        to_record=_hellaswag,
        n_sample=500,
        stratified=True,
        prompt_format="lm-eval-harness: '{activity_label}: {ctx_a} {ctx_b.capitalize()}' + ' {ending}'",
    ),
    TaskSpec(
        name="arc_easy",
        revision="210d026faf9955653af8916fad021475a3f00453",
        repo="allenai/ai2_arc",
        subset="ARC-Easy",
        split="test",
        language="en",
        to_record=_arc_easy,
        n_sample=500,
        stratified=False,
        prompt_format="lm-eval-harness: 'Question: {question}\\nAnswer:' + ' {choice}'",
    ),
    TaskSpec(
        name="kobest_hellaswag",
        revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
        repo="skt/kobest_v1",
        subset="hellaswag",
        split="test",
        language="ko",
        to_record=_kobest_hellaswag,
        n_sample=None,  # the split is exactly 500; take it whole
        stratified=False,
        prompt_format="'{context}' + ' {ending_N}'",
    ),
    # ---- English (added for the full post-training evaluation, spec 21.3) ----
    TaskSpec(
        name="piqa",
        revision="142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        repo="baber/piqa",  # parquet mirror the harness loads (no remote code)
        subset=None,
        split="validation",  # test ships label -1
        language="en",
        to_record=_piqa,
        n_sample=500,
        stratified=False,
        prompt_format="lm-eval-harness: 'Question: {goal}\\nAnswer:' + ' {sol}'",
    ),
    TaskSpec(
        name="arc_challenge",
        revision="210d026faf9955653af8916fad021475a3f00453",
        repo="allenai/ai2_arc",
        subset="ARC-Challenge",
        split="test",
        language="en",
        to_record=_arc_easy,  # identical schema to ARC-Easy
        n_sample=500,
        stratified=False,
        prompt_format="lm-eval-harness: 'Question: {question}\\nAnswer:' + ' {choice}'",
    ),
    TaskSpec(
        name="boolq",
        revision="3de24cf8022e94f4ee4b9d55a6f539891524d646",
        repo="aps/super_glue",
        subset="boolq",
        split="validation",  # test ships label -1
        language="en",
        to_record=_boolq,
        n_sample=500,
        stratified=False,
        prompt_format="lm-eval-harness: '{passage}\\nQuestion: {question}?\\nAnswer:' + ' {no|yes}'",
    ),
    TaskSpec(
        name="winogrande",
        revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        repo="allenai/winogrande",
        subset="winogrande_xl",
        split="validation",  # test ships an empty answer
        language="en",
        to_record=_winogrande,
        n_sample=500,
        stratified=False,
        prompt_format="lm-eval-harness partial: score '{suffix}' under '{prefix}{option_i}'",
    ),
    TaskSpec(
        name="openbookqa",
        revision="388097ea7776314e93a529163e0fea805b8a6454",
        repo="allenai/openbookqa",
        subset="main",
        split="test",
        language="en",
        to_record=_openbookqa,
        n_sample=500,  # the test split is exactly 500; taken whole
        stratified=False,
        prompt_format="lm-eval-harness: '{question_stem}' + ' {choice}'",
    ),
    # ---- Korean KoBEST suite (skt/kobest_v1) --------------------------------
    TaskSpec(
        name="kobest_copa",
        revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
        repo="skt/kobest_v1",
        subset="copa",
        split="test",
        language="ko",
        to_record=_kobest_copa,
        n_sample=500,
        stratified=False,
        prompt_format="'{premise} {왜냐하면|그래서}' + ' {alternative}'",
    ),
    TaskSpec(
        name="kobest_boolq",
        revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
        repo="skt/kobest_v1",
        subset="boolq",
        split="test",
        language="ko",
        to_record=_kobest_boolq,
        n_sample=500,
        stratified=False,
        prompt_format="'{paragraph} 질문: {question} 답변:' + ' {아니오|예}'",
    ),
    TaskSpec(
        name="kobest_sentineg",
        revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
        repo="skt/kobest_v1",
        subset="sentineg",
        split="test",
        language="ko",
        to_record=_kobest_sentineg,
        n_sample=None,  # the test split is 397; take it whole
        stratified=False,
        prompt_format="'문장: {sentence} 긍부정:' + ' {부정|긍정}'",
    ),
    TaskSpec(
        name="kobest_wic",
        revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
        repo="skt/kobest_v1",
        subset="wic",
        split="test",
        language="ko",
        to_record=_kobest_wic,
        n_sample=500,
        stratified=False,
        prompt_format="'문장1: {c1} 문장2: {c2} 두 문장에서 {word}가 같은 뜻으로 쓰였나?' + ' {아니오|예}'",
    ),
]

# LAMBADA would be the next addition: a formatter plus an entry in
# evaluation.downstream_tasks. It is the exception to the multiple-choice shape,
# being last-token accuracy (kkoma.evaluation.downstream has a separate scorer).


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample(records: list[dict], spec: TaskSpec, seed: int) -> list[dict]:
    """Pick the frozen subset. Deterministic given (seed, dataset revision)."""

    rng = random.Random(seed)
    n = spec.n_sample
    if n is None or n >= len(records):
        return list(records)

    if not spec.stratified:
        picked = rng.sample(records, n)
    else:
        # Equal count per gold label, so a model that collapses onto one answer
        # index cannot score above chance by accident.
        by_label: dict[int, list[dict]] = {}
        for r in records:
            by_label.setdefault(r["label"], []).append(r)
        labels = sorted(by_label)
        per = n // len(labels)
        short = [l for l in labels if len(by_label[l]) < per]
        if short:
            raise SystemExit(
                f"[{spec.name}] cannot stratify {n} across {len(labels)} labels: "
                f"label(s) {short} have fewer than {per} examples"
            )
        picked = []
        for label in labels:
            picked.extend(rng.sample(by_label[label], per))
        # n may not divide evenly; top up from what is left, still seeded.
        if len(picked) < n:
            chosen = {id(r) for r in picked}
            rest = [r for r in records if id(r) not in chosen]
            picked.extend(rng.sample(rest, n - len(picked)))

    # Break up the per-label grouping: under DDP the set is sharded across
    # ranks, and contiguous labels would hand each rank a skewed slice.
    rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Source cleanup
# ---------------------------------------------------------------------------


def _purge_source(repo: str) -> list[str]:
    """Delete one dataset's download. Scoped to `repo` by exact id."""

    removed: list[str] = []

    try:
        from datasets.config import HF_DATASETS_CACHE

        d = os.path.join(str(HF_DATASETS_CACHE), repo.replace("/", "___"))
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed.append(d)
    except Exception as exc:
        print(f"  ! datasets cache for {repo} not removed: {exc}")

    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        hits = [
            r for r in cache.repos
            if r.repo_id == repo and r.repo_type == "dataset"
        ]
        for r in hits:
            path = str(r.repo_path)
            cache.delete_revisions(
                *[rev.commit_hash for rev in r.revisions]
            ).execute()
            removed.append(path)
    except Exception as exc:
        print(f"  ! hub cache for {repo} not removed: {exc}")

    return removed


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(out_dir: str, seed: int, entries: list[dict], purged: list[str]) -> str:
    manifest = {
        "purpose": "frozen downstream evaluation sets (spec 21.3)",
        "sampling_seed": seed,
        "tasks": entries,
        "source_cache_removed": purged,
    }
    path = os.path.join(out_dir, "downstream_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_task(spec: TaskSpec, out_dir: str, seed: int) -> dict:
    import inspect

    from datasets import load_dataset

    subset = f":{spec.subset}" if spec.subset else ""
    t0 = section(f"{spec.name}  (downloading {spec.repo}{subset}, {spec.split})")

    # A few sources ship a loading script (winogrande, super_glue, hellaswag …).
    # datasets < 4.0 needs trust_remote_code=True to run it; datasets >= 5.0
    # removed the argument entirely and loads these via the Hub's auto-parquet
    # export instead. Pass the flag only when the installed version accepts it,
    # so we neither raise nor print the 5.0 "not supported anymore" notice.
    kwargs = {}
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        kwargs["trust_remote_code"] = True
    ds = load_dataset(
        spec.repo, spec.subset, split=spec.split, revision=spec.revision, **kwargs
    )
    source_size = len(ds)

    records, dropped = [], 0
    for i, row in enumerate(ds):
        rec = spec.to_record(row, i)
        if rec is None:
            dropped += 1
            continue
        records.append(rec)

    picked = _sample(records, spec, seed)
    path = os.path.join(out_dir, f"{spec.name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_choices = sorted({len(r["choices"]) for r in picked})
    labels = sorted({r["label"] for r in picked})
    line(
        f"{len(picked):,} of {source_size:,} kept  ·  choices={n_choices}  ·  labels={labels}"
        + (f"  ·  dropped {dropped} without gold" if dropped else "")
    )
    done(t0)

    return {
        "name": spec.name,
        "language": spec.language,
        "source": {
            "repo": spec.repo,
            "subset": spec.subset,
            "split": spec.split,
            "revision": spec.revision,
        },
        "source_size": source_size,
        "sample_size": len(picked),
        "dropped_without_gold": dropped,
        "sampling": (
            f"stratified by gold label ({spec.n_sample // 4} x 4)"
            if spec.stratified
            else ("whole split" if spec.n_sample is None else f"uniform random {spec.n_sample}")
        ),
        "sampling_seed": seed,
        "choice_counts": n_choices,
        "prompt_format": spec.prompt_format,
        "file": {"path": path, "sha256": _checksum(path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen downstream eval sets")
    parser.add_argument("--output-dir", default="data/downstream")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[t.name for t in TASKS],
        help=f"subset of: {' '.join(t.name for t in TASKS)}",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep the downloaded source datasets instead of deleting them",
    )
    args = parser.parse_args()

    known = {t.name: t for t in TASKS}
    unknown = [t for t in args.tasks if t not in known]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown} (known: {sorted(known)})")
    specs = [known[t] for t in args.tasks]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"preparing {len(specs)} downstream task(s) -> {args.output_dir} (seed {args.seed})")

    entries, purged = [], []
    for spec in specs:
        entries.append(build_task(spec, args.output_dir, args.seed))
        if not args.keep_source:
            purged.extend(_purge_source(spec.repo))

    manifest = _write_manifest(args.output_dir, args.seed, entries, purged)
    total = sum(e["sample_size"] for e in entries)
    if purged:
        line(f"removed {len(purged)} source cache path(s); pass --keep-source to retain them")
    ok(f"{total:,} questions across {len(entries)} task(s); manifest -> {manifest}")

    hard_exit()


if __name__ == "__main__":
    main()
