"""Tokenizer evaluation (spec section 4.6).

Computes compression / efficiency metrics for English and Korean and writes
``evaluation.json``. Korean efficiency is measured per eojeol (whitespace
unit); English per word.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from kkoma.tokenizer.special_tokens import SEMANTIC_TOKENS
from kkoma.tokenizer.utils import KkomaTokenizer

QUALITATIVE_SENTENCES = [
    "The model is trained from scratch.",
    "이 모델은 처음부터 직접 학습되었습니다.",
    "Kkoma-LLM은 영어와 한국어를 모두 처리합니다.",
    "def train_model(config):\n    return model",
]


@dataclass
class CorpusStats:
    n_docs: int = 0
    n_chars: int = 0
    n_words: int = 0
    n_tokens: int = 0
    # Tokens that decode to a single character. This is the over-fragmentation
    # signal: a tokenizer that never learned useful Korean merges falls back to
    # emitting one token per syllable, which shows up here.
    #
    # It replaces an earlier `n_byte_tokens`, which counted "<0xNN>" fallback
    # tokens and was structurally always zero: a ByteLevel pre-tokenizer maps
    # every byte into the alphabet, so byte_fallback never fires and no such
    # token can exist in this vocabulary (docs/audit-2026-07.md).
    n_single_char_tokens: int = 0

    def merge(self, other: "CorpusStats") -> None:
        self.n_docs += other.n_docs
        self.n_chars += other.n_chars
        self.n_words += other.n_words
        self.n_tokens += other.n_tokens
        self.n_single_char_tokens += other.n_single_char_tokens


def _is_single_char_token(tok: KkomaTokenizer, token_id: int) -> bool:
    """Does this id decode to exactly one character?

    Decoding rather than reading the raw token string, because ByteLevel
    represents a space as "Ġ" and multi-byte characters as several alphabet
    symbols — the raw string's length says nothing about the text it covers.
    """

    return len(tok.decode([token_id], skip_special_tokens=False)) == 1


def _score_text(tok: KkomaTokenizer, text: str) -> CorpusStats:
    ids = tok.encode(text)
    n_single = sum(1 for i in ids if _is_single_char_token(tok, i))
    return CorpusStats(
        n_docs=1,
        n_chars=len(text),
        n_words=len(text.split()),
        n_tokens=len(ids),
        n_single_char_tokens=n_single,
    )


def evaluate_corpus(tok: KkomaTokenizer, texts) -> dict:
    stats = CorpusStats()
    for text in texts:
        stats.merge(_score_text(tok, text))
    docs = max(stats.n_docs, 1)
    words = max(stats.n_words, 1)
    chars = max(stats.n_chars, 1)
    return {
        "documents": stats.n_docs,
        "tokens": stats.n_tokens,
        "tokens_per_word": stats.n_tokens / words,
        "tokens_per_char": stats.n_tokens / chars,
        "compression_ratio_char_per_token": chars / max(stats.n_tokens, 1),
        "avg_sequence_length": stats.n_tokens / docs,
        # Fraction of tokens covering a single character — rises when the
        # tokenizer has no useful merges for a script and falls back to
        # per-syllable output.
        "single_char_token_fraction": stats.n_single_char_tokens / max(stats.n_tokens, 1),
    }


def evaluate_tokenizer(
    tokenizer_path: str,
    english_texts,
    korean_texts,
    output_path: Optional[str] = None,
) -> dict:
    tok = KkomaTokenizer.from_file(tokenizer_path)

    result = {
        "vocab_size": len(tok),
        "embedding_parameters": len(tok),  # per-row; multiply by d_model downstream
        # Each special token must encode to exactly one id. Checking only that
        # it *has* an id (the previous check) would still pass if the token
        # were split into pieces by the pre-tokenizer.
        "special_tokens_single_id": {
            t: (tok.encode(t, allow_special=True) == [tok.token_to_id(t)])
            for t in SEMANTIC_TOKENS
        },
        "english": evaluate_corpus(tok, english_texts),
        "korean": evaluate_corpus(tok, korean_texts),
        "qualitative": [
            {
                "text": s,
                "ids": tok.encode(s),
                "n_tokens": len(tok.encode(s)),
                # Exact round trip. The previous check only asserted the
                # decode was non-empty, which a tokenizer that mangled every
                # Korean character would still satisfy.
                "roundtrip_ok": tok.decode(tok.encode(s), skip_special_tokens=False) == s,
            }
            for s in QUALITATIVE_SENTENCES
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


__all__ = [
    "evaluate_tokenizer",
    "evaluate_corpus",
    "QUALITATIVE_SENTENCES",
    "CorpusStats",
]
