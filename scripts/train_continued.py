"""Phase 3: Korean continued pretraining.

Loads a Kkoma-<size>-Base checkpoint and continues training on a Korean-heavy
mixture (default 70% KO / 30% EN replay, by document — see the note below).
``--init-from`` loads weights only, so the run starts with a fresh optimizer
state; ``--resume`` instead restores a full CPT run in progress.

Note the mixture weights are per *document*, not per token. Sources with longer
documents are over-represented per token and drain first: for the shipped 1.3B
run the realized split was 64/36 until the English replay ran out at ~84% of the
budget, after which the stream was Korean only. MixtureStream announces the
exhaustion when it happens.

Usage:
    python scripts/train_continued.py \
        --config configs/continued_pretraining/ko_800m.yaml \
        --init-from artifacts/checkpoints/base-800m/final.pt
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import RunConfig
from scripts._common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Kkoma Korean continued pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from", required=True, help="Base checkpoint to adapt")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    run_training(config, resume_path=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
