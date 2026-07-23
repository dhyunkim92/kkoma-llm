"""Phase 1: fixed corpus for controlled architecture comparison (spec 6.5).

Mixture target: FineWeb-Edu 95% + FineWeb2 Korean 5%. Default 300M train tokens
and 10M validation tokens at context length 1,024. Token counting requires the
trained tokenizer.

Documents are seed-shuffled while writing (spec 14.2) and train/validation are
separated by a document-level hash holdout (spec 14.3), so the two corpora are
disjoint by construction.

Each language is written to its own directory so the run config can re-mix them
at the target ratio at load time and compute EN/KO validation loss separately:

    data/architecture/train/      (English, 95% of the train budget)
    data/architecture/train_ko/   (Korean, 5% of the train budget)
    data/architecture/val/        (English validation)
    data/architecture/val_ko/     (Korean validation)

Usage:
    python scripts/prepare_architecture_data.py \
        --output-dir data/architecture --tokenizer artifacts/tokenizer
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import DataSource
from kkoma.tokenizer.utils import KkomaTokenizer
from scripts._console import ok
from scripts._prepare import hard_exit, prepare_corpus

EN = DataSource(name="HuggingFaceFW/fineweb-edu", subset="default", weight=1.0)
KO = DataSource(name="HuggingFaceFW/fineweb-2", subset="kor_Hang", weight=1.0)
EN_RATIO, KO_RATIO = 0.95, 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare architecture-study corpus")
    parser.add_argument("--output-dir", default="data/architecture")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--train-tokens", type=int, default=300_000_000)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-mod", type=int, default=100,
                        help="1/N of documents go to the validation pool")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000,
                        help="seed-shuffle buffer size (0 disables shuffling)")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = parser.parse_args()

    tok = KkomaTokenizer.from_file(args.tokenizer)
    show = not args.no_progress
    out = args.output_dir
    common = dict(tokenizer=tok, show_progress=show,
                  holdout_mod=args.holdout_mod, shuffle_buffer=args.shuffle_buffer)

    # Split the budget by the target ratio so both corpora deplete together.
    en_tr = prepare_corpus([EN], f"{out}/train", max_tokens=int(args.train_tokens * EN_RATIO),
                           seed=args.seed, split="train", **common)
    ko_tr = prepare_corpus([KO], f"{out}/train_ko", max_tokens=int(args.train_tokens * KO_RATIO),
                           seed=args.seed, split="train", **common)
    en_va = prepare_corpus([EN], f"{out}/val", max_tokens=args.val_tokens,
                           seed=args.seed + 1, split="val", **common)
    ko_va = prepare_corpus([KO], f"{out}/val_ko", max_tokens=args.val_tokens // 2,
                           seed=args.seed + 1, split="val", **common)

    ok(
        f"train: en={en_tr.tokens/1e6:.1f}M ko={ko_tr.tokens/1e6:.1f}M | "
        f"val: en={en_va.tokens/1e6:.1f}M ko={ko_va.tokens/1e6:.1f}M"
    )

    # Avoid the datasets-streaming shutdown race; data is already written.
    hard_exit()


if __name__ == "__main__":
    main()
