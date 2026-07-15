"""Phase 3: Korean continued-pretraining corpus (spec section 19.2).

Mixture target: FineWeb2 Korean 70% + English replay 30%. Default 2B tokens
(≈1.4B KO + 0.6B EN). Train and validation are separated by the same
document-level hash holdout used for the Base corpus (spec 14.3), so the
English replay and the English validation set cannot overlap.

Layout (matches configs/continued_pretraining/ko_800m.yaml):

    data/korean/train_ko/   (Korean, 70% of the train budget)
    data/korean/train_en/   (English replay, 30% of the train budget)
    data/korean/val_ko/     (Korean validation)
    data/korean/val_en/     (English validation, for forgetting measurement)

Usage:
    python scripts/prepare_korean_data.py \
        --output-dir data/korean --tokenizer artifacts/tokenizer --tokens 2e9
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import DataSource
from kkoma.tokenizer.utils import KkomaTokenizer
from scripts._prepare import hard_exit, prepare_corpus

KO = DataSource(name="HuggingFaceFW/fineweb-2", subset="kor_Hang", weight=1.0)
EN = DataSource(name="HuggingFaceFW/fineweb-edu", subset="default", weight=1.0)
KO_RATIO, EN_RATIO = 0.70, 0.30


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Korean continued-pretraining corpus")
    parser.add_argument("--output-dir", default="data/korean")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--tokens", type=float, default=2e9)
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
    tokens = int(args.tokens)
    common = dict(tokenizer=tok, show_progress=show,
                  holdout_mod=args.holdout_mod, shuffle_buffer=args.shuffle_buffer)

    ko_tr = prepare_corpus([KO], f"{out}/train_ko", max_tokens=int(tokens * KO_RATIO),
                           seed=args.seed, split="train", **common)
    en_tr = prepare_corpus([EN], f"{out}/train_en", max_tokens=int(tokens * EN_RATIO),
                           seed=args.seed, split="train", **common)
    ko_va = prepare_corpus([KO], f"{out}/val_ko", max_tokens=args.val_tokens,
                           seed=args.seed + 3, split="val", **common)
    en_va = prepare_corpus([EN], f"{out}/val_en", max_tokens=args.val_tokens // 2,
                           seed=args.seed + 3, split="val", **common)

    print(
        f"train: ko={ko_tr.tokens/1e9:.2f}B en={en_tr.tokens/1e9:.2f}B | "
        f"val: ko={ko_va.tokens/1e6:.1f}M en={en_va.tokens/1e6:.1f}M"
    )

    # Avoid the datasets-streaming shutdown race; data is already written.
    hard_exit()


if __name__ == "__main__":
    main()
