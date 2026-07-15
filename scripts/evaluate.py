"""Evaluate a checkpoint: LM loss (EN/KO), efficiency, generation samples.

    python scripts/evaluate.py \
        --checkpoint artifacts/checkpoints/base_125m/final.pt \
        --config configs/pretraining/base_125m.yaml \
        --output artifacts/evaluation/base_125m.json
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from kkoma.config import RunConfig
from kkoma.evaluation.efficiency import benchmark_forward
from kkoma.evaluation.generation import generate_samples
from kkoma.evaluation.language_modeling import evaluate_bilingual
from kkoma.model.model import KkomaModel
from kkoma.training.checkpoint import load_checkpoint
from scripts._common import build_tokenizer, build_val_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Kkoma checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/evaluation/eval.json")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KkomaModel(config.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    tokenizer = build_tokenizer(config)
    results: dict = {"checkpoint": args.checkpoint, "parameters": model.parameter_report()}

    val_loaders = build_val_loaders(config, tokenizer)
    if val_loaders:
        results["language_modeling"] = evaluate_bilingual(
            model, val_loaders, device, max_batches=args.max_batches
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
