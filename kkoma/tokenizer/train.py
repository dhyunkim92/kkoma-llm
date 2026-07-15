"""Train the Kkoma byte-level BPE tokenizer (spec section 4).

The BPE algorithm itself comes from the Hugging Face ``tokenizers`` Rust
backend; only the Kkoma vocabulary and merge rules are learned here. The
trainer's ``vocab_size`` includes the special tokens, so the final tokenizer
length is exactly 32,768.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from kkoma.tokenizer.special_tokens import SPECIAL_TOKENS, special_tokens_map


@dataclass
class TokenizerTrainConfig:
    vocab_size: int = 32768
    min_frequency: int = 2
    output_dir: str = "artifacts/tokenizer"
    # Files are plain-text or JSONL; one document per record.
    input_files: list[str] = field(default_factory=list)
    text_key: str = "text"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str, input_files: list[str]) -> "TokenizerTrainConfig":
        """Build from ``configs/tokenizer/*.yaml``.

        Consumes ``vocab_size`` / ``min_frequency`` / ``output_dir`` / ``seed``;
        the ``data`` section belongs to ``prepare_tokenizer_data.py``. Special
        tokens are not configurable: their identity and order are fixed in
        ``kkoma.tokenizer.special_tokens`` (later phases depend on stable ids).
        """

        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {"vocab_size", "min_frequency", "output_dir", "seed"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        unknown = set(raw) - known - {"data"}
        if unknown:
            raise ValueError(f"unknown tokenizer config key(s) in {path}: {sorted(unknown)}")
        return cls(input_files=input_files, **kwargs)


def _iter_text(files: Iterable[str], text_key: str) -> Iterator[str]:
    for path in files:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get(text_key, "")
                    if text:
                        yield text
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n")


def train_tokenizer(config: TokenizerTrainConfig):
    """Train and save the tokenizer artifacts; returns the Tokenizer object."""

    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    # Byte-level: full byte coverage, no UNK needed; minimal normalization.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _iter_text(config.input_files, config.text_key), trainer=trainer
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    _verify_and_save(tokenizer, config)
    return tokenizer


def _verify_and_save(tokenizer, config: TokenizerTrainConfig) -> None:
    size = tokenizer.get_vocab_size()
    if size != config.vocab_size:
        raise ValueError(
            f"final tokenizer length {size} != target {config.vocab_size}; "
            "check that special tokens are counted within vocab_size"
        )
    # Every special token must map to a single id.
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        if tid is None:
            raise ValueError(f"special token not found: {tok}")

    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(config.output_dir, "tokenizer.json"))
    tokenizer.model.save(config.output_dir)  # vocab.json + merges.txt

    with open(
        os.path.join(config.output_dir, "special_tokens_map.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(special_tokens_map(), f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(config.output_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 1024,
                **special_tokens_map(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        os.path.join(config.output_dir, "training_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "vocab_size": config.vocab_size,
                "min_frequency": config.min_frequency,
                "seed": config.seed,
                "special_tokens": SPECIAL_TOKENS,
                "input_files": config.input_files,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


__all__ = ["TokenizerTrainConfig", "train_tokenizer"]
