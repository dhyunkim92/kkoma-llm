"""Leaderboard aggregation (scripts/leaderboard.py): ranking, columns, gaps."""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.leaderboard import collect_rows, render_markdown, write_csv


def _write_eval(path, *, run_name, overall, tasks, params=1_000_000, val_loss=3.0):
    data = {
        "run_name": run_name,
        "checkpoint": f"artifacts/checkpoints/{run_name}/final.pt",
        "parameters": {"total": params},
        "language_modeling": {"all": {"loss": val_loss, "perplexity": 20.0}},
        "downstream": {
            "tasks": {name: {"language": "en", "acc_norm": acc} for name, acc in tasks.items()},
            "aggregates": {"en_avg": overall, "ko_avg": overall, "overall_avg": overall},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_rows_ranked_by_overall_descending(tmp_path):
    a = _write_eval(tmp_path / "a.json", run_name="small", overall=0.30, tasks={"piqa": 0.30})
    b = _write_eval(tmp_path / "b.json", run_name="big", overall=0.45, tasks={"piqa": 0.45})
    rows, task_names = collect_rows([a, b])
    assert [r["model"] for r in rows] == ["big", "small"]  # highest overall first
    assert task_names == ["piqa"]


def test_task_columns_are_union_and_missing_render_as_dash(tmp_path):
    a = _write_eval(tmp_path / "a.json", run_name="m1", overall=0.4, tasks={"piqa": 0.4})
    b = _write_eval(tmp_path / "b.json", run_name="m2", overall=0.5, tasks={"boolq": 0.6})
    rows, task_names = collect_rows([a, b])
    assert task_names == ["boolq", "piqa"]  # sorted union
    md = render_markdown(rows, task_names)
    # m1 has no boolq, m2 has no piqa -> each shows a dash for the missing task.
    assert md.count("–") == 2
    assert "boolq | piqa" in md  # header carries both task columns


def test_run_without_downstream_sorts_last(tmp_path):
    scored = _write_eval(tmp_path / "a.json", run_name="scored", overall=0.4, tasks={"piqa": 0.4})
    bare = tmp_path / "b.json"
    bare.write_text(json.dumps({
        "run_name": "loss-only",
        "parameters": {"total": 2_000_000},
        "language_modeling": {"all": {"loss": 2.5, "perplexity": 12.0}},
    }), encoding="utf-8")
    rows, _ = collect_rows([str(bare), scored])
    assert [r["model"] for r in rows] == ["scored", "loss-only"]  # missing overall last
    assert rows[-1]["overall_avg"] is None


def test_csv_has_raw_numbers(tmp_path):
    a = _write_eval(tmp_path / "a.json", run_name="m1", overall=0.42, tasks={"piqa": 0.42}, params=879_000_000)
    rows, task_names = collect_rows([a])
    out = tmp_path / "lb.csv"
    write_csv(rows, task_names, str(out))
    with open(out, encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert got[0]["model"] == "m1"
    assert float(got[0]["overall_avg"]) == 0.42  # raw fraction, not a percent string
    assert int(got[0]["params"]) == 879_000_000
    assert float(got[0]["piqa"]) == 0.42


def test_markdown_formats_percent_and_params(tmp_path):
    a = _write_eval(tmp_path / "a.json", run_name="base-1b", overall=0.412, tasks={"piqa": 0.5}, params=879_000_000)
    rows, task_names = collect_rows([a])
    md = render_markdown(rows, task_names)
    assert "41.2" in md      # overall as a percent, one decimal
    assert "879.0M" in md    # params in millions
    assert "base-1b" in md
