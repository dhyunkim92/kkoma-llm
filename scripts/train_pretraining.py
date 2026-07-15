"""Phase 2: Base pretraining.

Usage:
    python scripts/train_pretraining.py --config configs/pretraining/base_125m.yaml
    # multi-GPU:
    torchrun --nproc_per_node=8 scripts/train_pretraining.py --config <cfg> [--resume <ckpt>]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import RunConfig
from scripts._common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Kkoma Base pretraining")
    parser.add_argument("--config", required=True, help="YAML run config path")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    run_training(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
