"""Evaluate a checkpoint: LM loss (EN/KO), downstream benchmarks, efficiency,
generation samples.

    python scripts/evaluate.py \
        --checkpoint artifacts/checkpoints/base-125m/final.pt \
        --config configs/pretraining/base_125m.yaml \
        --output artifacts/evaluation/base_125m.json

This is the post-training evaluation run (spec 21.3): it scores every enabled
downstream task in the config, not just the light set the trainer tracks during
training. Prepare the frozen sets first with scripts/prepare_downstream_data.py;
pass --no-downstream to skip them (e.g. a quick LM-loss-only check).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from kkoma.config import RunConfig
from kkoma.evaluation.downstream import evaluate_downstream_suite
from kkoma.evaluation.efficiency import benchmark_forward
from kkoma.evaluation.generation import generate_samples
from kkoma.evaluation.language_modeling import evaluate_bilingual
from kkoma.model.model import KkomaModel
from kkoma.training.checkpoint import load_checkpoint
from scripts._common import build_downstream_tasks, build_tokenizer, build_val_loaders
from scripts._console import done, line, ok, section


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Kkoma checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/evaluation/eval.json")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-downstream", action="store_true",
                        help="skip the downstream benchmark pass")
    parser.add_argument("--val-from", metavar="CONFIG", default=None,
                        help="take data.val_sources from this config instead of --config, "
                             "so several models can be scored on one fixed corpus")
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    # A model's own config names the corpus it trained against, which is not
    # necessarily one that still exists or that its peers were measured on.
    # Comparing val losses across corpora is meaningless, so allow pinning the
    # evaluation corpus independently of the training config.
    if args.val_from:
        config.data.val_sources = RunConfig.from_yaml(args.val_from).data.val_sources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = section(f"loading checkpoint on {device}")
    model = KkomaModel(config.model).to(device)
    # map_location="cpu": a training checkpoint carries the AdamW state (two
    # moments, ~2x the weights) that evaluation never touches. Mapping straight
    # to CUDA would pull all of it onto the GPU before it is discarded — 17.1 GB
    # peak instead of 4.5 GB on the 1.3B. load_state_dict copies the CPU tensors
    # into the already-placed CUDA parameters.
    # restore_rng=False: evaluation only reads the model; it must not adopt the
    # training run's RNG state (generation uses evaluation.sampling_seed instead).
    load_checkpoint(args.checkpoint, model, map_location="cpu", restore_rng=False)
    model.eval()
    tokenizer = build_tokenizer(config)
    params = model.parameter_report()
    line(f"{config.project.run_name} | {params['total'] / 1e6:.1f}M params | {args.checkpoint}")
    done(t0)

    results: dict = {
        "run_name": config.project.run_name,  # leaderboard label (scripts/leaderboard.py)
        "checkpoint": args.checkpoint,
        "parameters": params,
        # Which data these numbers came from. Without this a leaderboard row is
        # not interpretable: two models evaluated on different validation
        # corpora are not comparable, and the config that names the corpus can
        # be edited (or its files deleted) long after the JSON is written.
        "provenance": {
            "config": args.config,
            "val_sources": [s.path for s in config.data.val_sources],
            "max_batches": args.max_batches,
            "downstream_paths": [
                t.path for t in config.evaluation.downstream_tasks if t.enabled
            ],
        },
    }

    # ---- validation loss (EN / KO) ----------------------------------------
    val_loaders = build_val_loaders(config, tokenizer)
    if val_loaders:
        t0 = section("language modeling (validation loss)")
        lm = evaluate_bilingual(model, val_loaders, device, max_batches=args.max_batches)
        results["language_modeling"] = lm
        for lang in [k for k in lm if k != "all"] + (["all"] if "all" in lm else []):
            r = lm[lang]
            line(f"{lang:<3} loss {r['loss']:.4f}  ppl {r['perplexity']:.1f}")
        done(t0)

    # ---- downstream suite: every enabled task (during_training_only=False) --
    if not args.no_downstream:
        tasks = build_downstream_tasks(config, during_training_only=False)
        if tasks:
            t0 = section(f"downstream benchmarks ({len(tasks)} tasks)")

            def _on_task(name, language, r):
                line(f"{name:<20} [{language}] acc_norm {r['acc_norm']:.3f}"
                      f"  chance {r['chance']:.3f}  above {r['above_chance']:+.3f}"
                      f"  ±2σ {2 * r['se']:.3f}  (n={r['n']})")

            results["downstream"] = evaluate_downstream_suite(
                model, tokenizer, tasks, device,
                batch_size=config.evaluation.downstream_batch_size,
                on_task=_on_task,
            )
            agg = results["downstream"]["aggregates"]
            for lang in ("en", "ko", "overall"):
                if f"{lang}_avg" in agg:
                    line(f"→ {lang:<7} avg {agg[f'{lang}_avg']:.3f}"
                         f"  (chance {agg[f'{lang}_chance']:.3f},"
                         f" above {agg[f'{lang}_above_chance']:+.3f})")
            done(t0)
        else:
            print("\n▸ downstream benchmarks … skipped")
            line("no benchmark files found; run scripts/prepare_downstream_data.py "
                  "(or pass --no-downstream)")

    # ---- efficiency -------------------------------------------------------
    t0 = section("efficiency")
    eff = benchmark_forward(
        model, device, batch_size=2, seq_len=min(512, config.model.context_length), steps=5
    )
    results["efficiency"] = eff
    line(f"{eff['tokens_per_second']:,.0f} tok/s  |  {eff['step_time_ms']:.1f} ms/step"
          f"  |  {eff['peak_allocated_mb']:,.0f} MB peak")
    done(t0)

    # ---- generation samples (fixed prompts, spec 21.4) --------------------
    if not args.no_generation:
        ev = config.evaluation
        prompts = ev.generation_prompts_en + ev.generation_prompts_ko
        t0 = section(f"generation ({len(prompts)} fixed prompts)")
        samples = generate_samples(
            model, tokenizer, device, prompts=prompts, seed=ev.sampling_seed,
        )
        results["generation"] = samples
        for s in samples:
            cont = s["completion"].replace("\n", " ")
            if len(cont) > 80:
                cont = cont[:80] + "…"
            line(f"{s['prompt']!r} → {cont}")
        done(t0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    ok(f"evaluation written to {args.output}")


if __name__ == "__main__":
    main()
