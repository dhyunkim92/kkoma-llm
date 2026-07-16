"""Phase 2: build the Base pretraining corpus (spec section 14.2).

Mixture target: FineWeb-Edu 95% + FineWeb2 Korean 5%. A seed-shuffled corpus is
produced per language; the run config re-mixes them at load time. Downstream the
125M / 350M / 800M / 1B / 1.3B models read fewer tokens of the same shards
(1.25B / 3.5B / 8B / 9B / 11.5B).

Train and validation are separated by a document-level hash holdout
(spec section 14.3): sha256(text) assigns ~1/holdout_mod of documents to the
validation pool, so the two corpora are disjoint by construction.

Layout (matches configs/pretraining/*.yaml):

    data/pretrain/train/      (English, 95% of the train budget)
    data/pretrain/train_ko/   (Korean, 5% of the train budget)
    data/pretrain/val/        (English validation)
    data/pretrain/val_ko/     (Korean validation)

Usage:
    python scripts/prepare_pretraining_data.py \
        --output-dir data/pretrain --tokenizer artifacts/tokenizer --tokens 11.5e9
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import DataSource
from kkoma.tokenizer.utils import KkomaTokenizer
from scripts._prepare import hard_exit, prepare_corpus

EN = DataSource(name="HuggingFaceFW/fineweb-edu", subset="default", weight=1.0)
KO = DataSource(name="HuggingFaceFW/fineweb-2", subset="kor_Hang", weight=1.0)
EN_RATIO, KO_RATIO = 0.95, 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Base pretraining corpus")
    parser.add_argument("--output-dir", default="data/pretrain")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--tokens", type=float, default=11.5e9, help="train token budget")
    parser.add_argument("--val-tokens", type=int, default=20_000_000)
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

    en_tr = prepare_corpus([EN], f"{out}/train", max_tokens=int(tokens * EN_RATIO),
                           seed=args.seed, split="train", **common)
    ko_tr = prepare_corpus([KO], f"{out}/train_ko", max_tokens=int(tokens * KO_RATIO),
                           seed=args.seed, split="train", **common)
    en_va = prepare_corpus([EN], f"{out}/val", max_tokens=args.val_tokens,
                           seed=args.seed + 7, split="val", **common)
    ko_va = prepare_corpus([KO], f"{out}/val_ko", max_tokens=args.val_tokens // 2,
                           seed=args.seed + 7, split="val", **common)

    print(
        f"train: en={en_tr.tokens/1e9:.2f}B ko={ko_tr.tokens/1e9:.2f}B | "
        f"val: en={en_va.tokens/1e6:.1f}M ko={ko_va.tokens/1e6:.1f}M"
    )
    print("125M/350M/800M/1B/1.3B read 1.25B/3.5B/8B/9B/11.5B of these shards (set by max_tokens in each config).")

    # Avoid the datasets-streaming shutdown race; data is already written.
    hard_exit()


if __name__ == "__main__":
    main()
