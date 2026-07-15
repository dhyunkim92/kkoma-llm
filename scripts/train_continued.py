"""Phase 3: Korean continued pretraining.

Loads the English Kkoma-800M-Base weights and continues training on a
Korean-heavy mixture (default 70% KO / 30% EN replay). The v1 default starts
with a fresh optimizer state (controlled by --fresh-optimizer / config).

Usage:
    python scripts/train_continued.py \
        --config configs/continued_pretraining/ko_800m.yaml \
        --init-from artifacts/checkpoints/base_800m/final.pt
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
