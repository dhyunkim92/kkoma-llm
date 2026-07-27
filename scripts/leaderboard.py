"""Aggregate scripts/evaluate.py outputs into a leaderboard (Markdown + CSV).

Scans a directory of per-checkpoint evaluation JSONs (one per model, written by
scripts/evaluate.py) and produces a single ranked table: one row per model,
columns for each downstream task's ``acc_norm``, the en/ko/overall aggregates,
and validation loss / perplexity. Ranked by overall downstream accuracy.

    # evaluate each checkpoint first
    python scripts/evaluate.py --checkpoint .../base-1b/final.pt  --config ... \
        --output artifacts/evaluation/base_1b.json
    python scripts/evaluate.py --checkpoint .../base-350m/final.pt --config ... \
        --output artifacts/evaluation/base_350m.json
    # then aggregate them
    python scripts/leaderboard.py \
        --input-dir artifacts/evaluation \
        --output artifacts/evaluation/leaderboard

Writes ``<output>.md`` (a table to paste into a README) and ``<output>.csv``
(raw numbers for a spreadsheet). Missing values render as ``–`` in Markdown and
stay empty in the CSV, so a run evaluated with ``--no-downstream`` still lists.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
import json

from scripts._console import ok


def _model_label(data: dict, path: str) -> str:
    """Row label: the run name, else the checkpoint's dir/stem, else the file."""

    if data.get("run_name"):
        return data["run_name"]
    ckpt = data.get("checkpoint")
    if ckpt:
        parent = os.path.basename(os.path.dirname(ckpt))
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        return f"{parent}/{stem}" if parent else stem
    return os.path.splitext(os.path.basename(path))[0]


def collect_rows(json_paths: list[str]) -> tuple[list[dict], list[str]]:
    """Read each evaluation JSON into a row; return (rows, sorted task names).

    Rows are ranked by ``overall_avg`` descending, with missing scores last.
    """

    rows: list[dict] = []
    task_names: set[str] = set()
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ds = data.get("downstream") or {}
        tasks = {name: t.get("acc_norm") for name, t in (ds.get("tasks") or {}).items()}
        task_names.update(tasks)
        agg = ds.get("aggregates") or {}
        lm_all = (data.get("language_modeling") or {}).get("all") or {}
        prov = data.get("provenance") or {}
        rows.append({
            "model": _model_label(data, path),
            "params": (data.get("parameters") or {}).get("total"),
            "tasks": tasks,
            "en_avg": agg.get("en_avg"),
            "ko_avg": agg.get("ko_avg"),
            "overall_avg": agg.get("overall_avg"),
            # Chance-corrected views. Raw averages mix 2-way and 4-way tasks, so
            # they are not comparable across languages; these are.
            "en_above": agg.get("en_above_chance"),
            "ko_above": agg.get("ko_above_chance"),
            "overall_above": agg.get("overall_above_chance"),
            "val_loss": lm_all.get("loss"),
            "val_ppl": lm_all.get("perplexity"),
            # Which validation corpus produced val_loss. Rows measured on
            # different corpora are not comparable and must not be read as a
            # ranking; render_markdown flags that case.
            "val_sources": tuple(prov.get("val_sources") or ()),
            "max_batches": prov.get("max_batches"),
        })

    # Rank by overall accuracy, highest first; rows without a score sort last.
    rows.sort(key=lambda r: (r["overall_avg"] is not None, r["overall_avg"] or 0.0), reverse=True)
    return rows, sorted(task_names)


def val_corpus_groups(rows: list[dict]) -> dict:
    """Map each distinct validation corpus to the models measured on it.

    A leaderboard whose ``val loss`` column spans more than one group is
    comparing models on different data; the numbers in it cannot be ranked
    against each other.
    """

    groups: dict = {}
    for r in rows:
        key = (r["val_sources"], r["max_batches"])
        groups.setdefault(key, []).append(r["model"])
    return groups


def _pct(x) -> str:
    return "–" if x is None else f"{x * 100:.1f}"


def _num(x, digits: int) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def render_markdown(rows: list[dict], task_names: list[str]) -> str:
    header = (
        ["Model", "Params", "Overall", "EN avg", "KO avg", "EN +chance", "KO +chance"]
        + task_names + ["val loss", "ppl"]
    )
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        params = "–" if r["params"] is None else f"{r['params'] / 1e6:.1f}M"
        cells = [r["model"], params, _pct(r["overall_avg"]), _pct(r["en_avg"]), _pct(r["ko_avg"]),
                 _pct(r["en_above"]), _pct(r["ko_above"])]
        cells += [_pct(r["tasks"].get(t)) for t in task_names]
        cells += [_num(r["val_loss"], 3), _num(r["val_ppl"], 1)]
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"

    # A split validation corpus makes the val-loss column meaningless as a
    # ranking. Say so in the artifact rather than leaving the reader to notice.
    groups = val_corpus_groups(rows)
    if len(groups) > 1:
        out += (
            "\n> **⚠ `val loss` is not comparable across these rows** — they were measured on\n"
            "> different validation corpora or sample sizes:\n>\n"
        )
        for (srcs, mb), models in groups.items():
            src = ", ".join(srcs) if srcs else "(not recorded)"
            out += f"> - `{src}` @ max_batches={mb}: {', '.join(models)}\n"
    return out


def write_csv(rows: list[dict], task_names: list[str], path: str) -> None:
    cols = (["model", "params", "overall_avg", "en_avg", "ko_avg",
             "en_above_chance", "ko_above_chance", "overall_above_chance"]
            + task_names + ["val_loss", "val_ppl", "val_sources", "max_batches"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [r["model"], r["params"], r["overall_avg"], r["en_avg"], r["ko_avg"],
                 r["en_above"], r["ko_above"], r["overall_above"]]
                + [r["tasks"].get(t) for t in task_names]
                + [r["val_loss"], r["val_ppl"], ";".join(r["val_sources"]), r["max_batches"]]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation JSONs into a leaderboard")
    parser.add_argument("--input-dir", default="artifacts/evaluation",
                        help="directory of scripts/evaluate.py output JSONs")
    parser.add_argument("--output", default="artifacts/evaluation/leaderboard",
                        help="output path stem; writes <stem>.md and <stem>.csv")
    parser.add_argument("--glob", default="*.json", help="which files to read in --input-dir")
    args = parser.parse_args()

    # Skip a previously written leaderboard if it happens to be JSON-adjacent.
    paths = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, args.glob))
        if not os.path.basename(p).startswith("leaderboard")
    )
    if not paths:
        raise SystemExit(f"no evaluation JSONs matching {args.glob} in {args.input_dir}")

    rows, task_names = collect_rows(paths)

    stem = args.output[:-3] if args.output.endswith(".md") else args.output
    md_path, csv_path = stem + ".md", stem + ".csv"
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    table = render_markdown(rows, task_names)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Kkoma downstream leaderboard\n\n")
        f.write("Downstream `acc_norm` (%), ranked by overall. "
                "Built by `scripts/leaderboard.py` from `scripts/evaluate.py` outputs.\n\n")
        f.write(table)
    write_csv(rows, task_names, csv_path)

    print(table)
    ok(f"wrote {md_path} and {csv_path} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
