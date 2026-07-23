"""Evaluate a checkpoint: LM loss (EN/KO), downstream benchmarks, efficiency,
generation samples.

    python scripts/evaluate.py \
        --checkpoint artifacts/checkpoints/base_125m/final.pt \
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Kkoma checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/evaluation/eval.json")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-downstream", action="store_true",
                        help="skip the downstream benchmark pass")
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KkomaModel(config.model).to(device)
    # restore_rng=False: evaluation only reads the model; it must not adopt the
    # training run's RNG state (generation uses evaluation.sampling_seed instead).
    load_checkpoint(args.checkpoint, model, map_location=str(device), restore_rng=False)
    model.eval()

    tokenizer = build_tokenizer(config)
    results: dict = {
        "run_name": config.project.run_name,  # leaderboard label (scripts/leaderboard.py)
        "checkpoint": args.checkpoint,
        "parameters": model.parameter_report(),
    }

    val_loaders = build_val_loaders(config, tokenizer)
    if val_loaders:
        results["language_modeling"] = evaluate_bilingual(
            model, val_loaders, device, max_batches=args.max_batches
        )

    # Full downstream suite: every enabled task (during_training_only=False),
    # not just the light set the trainer scores mid-run.
    if not args.no_downstream:
        tasks = build_downstream_tasks(config, during_training_only=False)
        if tasks:
            results["downstream"] = evaluate_downstream_suite(
                model, tokenizer, tasks, device,
                batch_size=config.evaluation.downstream_batch_size,
            )
            agg = results["downstream"]["aggregates"]
            print(
                "downstream: "
                + " ".join(f"{k}={v:.3f}" for k, v in agg.items())
            )
        else:
            print(
                "[downstream] no benchmark files found; run "
                "scripts/prepare_downstream_data.py (or pass --no-downstream)"
            )

    results["efficiency"] = benchmark_forward(
        model, device, batch_size=2, seq_len=min(512, config.model.context_length), steps=5
    )

    if not args.no_generation:
        # Fixed prompt set + sampling seed from the config (spec 21.4) so every
        # checkpoint is sampled under identical conditions.
        ev = config.evaluation
        results["generation"] = generate_samples(
            model, tokenizer, device,
            prompts=ev.generation_prompts_en + ev.generation_prompts_ko,
            seed=ev.sampling_seed,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"evaluation written to {args.output}")


if __name__ == "__main__":
    main()
