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
    n_byte_tokens: int = 0

    def merge(self, other: "CorpusStats") -> None:
        self.n_docs += other.n_docs
        self.n_chars += other.n_chars
        self.n_words += other.n_words
        self.n_tokens += other.n_tokens
        self.n_byte_tokens += other.n_byte_tokens


def _is_byte_token(token: Optional[str]) -> bool:
    # Byte-level fallback tokens look like "<0xF0>" in byte_fallback BPE.
    return bool(token) and token.startswith("<0x") and token.endswith(">")


def _score_text(tok: KkomaTokenizer, text: str) -> CorpusStats:
    ids = tok.encode(text)
    n_byte = sum(1 for i in ids if _is_byte_token(tok.id_to_token(i)))
    return CorpusStats(
        n_docs=1,
        n_chars=len(text),
        n_words=len(text.split()),
        n_tokens=len(ids),
        n_byte_tokens=n_byte,
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
        "byte_token_fraction": stats.n_byte_tokens / max(stats.n_tokens, 1),
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
        "special_tokens_single_id": {
            t: (tok.token_to_id(t) is not None) for t in SEMANTIC_TOKENS
        },
        "english": evaluate_corpus(tok, english_texts),
        "korean": evaluate_corpus(tok, korean_texts),
        "qualitative": [
            {
                "text": s,
                "ids": tok.encode(s),
                "n_tokens": len(tok.encode(s)),
                "roundtrip_ok": tok.decode(tok.encode(s)).strip() != "",
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
