"""Phase 0: sample the tokenizer training corpus (spec section 4.2).

Default mixture: English FineWeb-Edu 75% + Korean FineWeb2 25%, ~20GB total
(≈15GB EN + 5GB KO) measured by cleaned bytes. The mixture can also come from
the ``data`` section of the tokenizer config; CLI flags override it.

    python scripts/prepare_tokenizer_data.py \
        --config configs/tokenizer/tokenizer_32k.yaml --output-dir data/tokenizer
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kkoma.config import DataSource
from scripts._prepare import hard_exit, prepare_corpus


def default_sources() -> list[DataSource]:
    return [
        DataSource(name="HuggingFaceFW/fineweb-edu", subset="default", weight=0.75, text_key="text"),
        DataSource(name="HuggingFaceFW/fineweb-2", subset="kor_Hang", weight=0.25, text_key="text"),
    ]


def sources_from_config(path: str) -> tuple[list[DataSource], float]:
    """Read the mixture from a tokenizer YAML's ``data`` section.

    Returns the sources (weights from the config) and the total target size in
    GB (sum of per-language ``target_gb``).
    """

    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = (yaml.safe_load(f) or {}).get("data") or {}
    if not data:
        raise SystemExit(f"no data section in {path}")
    sources, total_gb = [], 0.0
    for lang, entry in data.items():
        sources.append(
            DataSource(
                name=entry["name"],
                subset=entry.get("subset"),
                weight=float(entry.get("weight", 1.0)),
            )
        )
        total_gb += float(entry.get("target_gb", 0.0))
    return sources, total_gb


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer training data")
    parser.add_argument("--config", default=None,
                        help="tokenizer YAML with a data section, e.g. configs/tokenizer/tokenizer_32k.yaml")
    parser.add_argument("--output-dir", default="data/tokenizer")
    parser.add_argument("--gb", type=float, default=None, help="target size in GB (overrides config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000,
                        help="seed-shuffle buffer size (0 disables shuffling)")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = parser.parse_args()

    if args.config:
        sources, config_gb = sources_from_config(args.config)
    else:
        sources, config_gb = default_sources(), 20.0
    gb = args.gb if args.gb is not None else config_gb

    # split="all": tokenizer data needs seed-based sampling (spec 4.3) but no
    # train/val holdout.
    result = prepare_corpus(
        sources,
        args.output_dir,
        max_bytes=int(gb * 1e9),
        seed=args.seed,
        show_progress=not args.no_progress,
        shuffle_buffer=args.shuffle_buffer,
    )
    print(
        f"docs={result.documents} bytes={result.bytes/1e9:.2f}GB "
        f"shards={len(result.shards)} manifest={result.manifest_path}"
    )

    # Avoid the datasets-streaming shutdown race; data is already written.
    hard_exit()


if __name__ == "__main__":
    main()
