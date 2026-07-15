"""Phase 1: Architecture Study run.

Trains one architecture variant of the progressive sets (baseline -> rope ->
rope_rmsnorm -> rope_rmsnorm_swiglu -> modern, at 125M and 350M) under the
shared controlled conditions. Only the model section of the config changes
between runs; everything else (tokenizer, corpus, budget, optimizer, seed)
stays fixed.

Usage:
    python scripts/train_architecture.py --config configs/architecture/core_125m_modern.yaml
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import RunConfig
from scripts._common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Kkoma architecture study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    run_training(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
