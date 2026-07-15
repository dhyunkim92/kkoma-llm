"""Phase 0: train and evaluate the Kkoma byte-level BPE tokenizer.

    python scripts/train_tokenizer.py \
        --input "data/tokenizer/*.jsonl" \
        --config configs/tokenizer/tokenizer_32k.yaml

CLI flags override the config file; without --config the built-in defaults
(vocab 32,768, min_frequency 2) apply.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

from kkoma.tokenizer.evaluate import evaluate_tokenizer
from kkoma.tokenizer.train import TokenizerTrainConfig, train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Kkoma tokenizer")
    parser.add_argument("--input", required=True, help="glob of jsonl/txt files")
    parser.add_argument("--config", default=None,
                        help="tokenizer YAML, e.g. configs/tokenizer/tokenizer_32k.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"no input files matched: {args.input}")

    config = (
        TokenizerTrainConfig.from_yaml(args.config, input_files=files)
        if args.config
        else TokenizerTrainConfig(input_files=files)
    )
    for arg_name, field_name in [
        ("output_dir", "output_dir"),
        ("vocab_size", "vocab_size"),
        ("min_frequency", "min_frequency"),
        ("seed", "seed"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)
    tokenizer = train_tokenizer(config)
    print(f"trained tokenizer: vocab_size={tokenizer.get_vocab_size()}")

    # Quick qualitative evaluation against the spec's sample sentences.
    evaluate_tokenizer(
        config.output_dir,
        english_texts=["The model is trained from scratch."],
        korean_texts=["이 모델은 처음부터 직접 학습되었습니다."],
        output_path=f"{config.output_dir}/evaluation.json",
    )
    print(f"evaluation written to {config.output_dir}/evaluation.json")


if __name__ == "__main__":
    main()
