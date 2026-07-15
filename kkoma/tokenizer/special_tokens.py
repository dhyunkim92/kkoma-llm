"""Kkoma special-token definitions (spec section 4.4).

7 meaning-fixed tokens + 16 future-expansion reserved tokens = 23 total. The
role tokens (system/user/assistant/turn_end) are included in the vocabulary
now but carry no learned meaning during Kkoma-LLM pretraining; they exist so
the same tokenizer serves Kkoma-Chat and Kkoma-Agent unchanged.
"""

from __future__ import annotations

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TURN_END = "<|turn_end|>"

# Order is fixed and must not change once a tokenizer is trained.
SEMANTIC_TOKENS = [BOS, EOS, PAD, SYSTEM, USER, ASSISTANT, TURN_END]

NUM_RESERVED = 16
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]

SPECIAL_TOKENS = SEMANTIC_TOKENS + RESERVED_TOKENS

assert len(SEMANTIC_TOKENS) == 7
assert len(RESERVED_TOKENS) == 16
assert len(SPECIAL_TOKENS) == 23


def special_tokens_map() -> dict[str, str]:
    """Mapping used by HF tokenizer_config / special_tokens_map.json."""

    return {
        "bos_token": BOS,
        "eos_token": EOS,
        "pad_token": PAD,
        "additional_special_tokens": [
            SYSTEM,
            USER,
            ASSISTANT,
            TURN_END,
            *RESERVED_TOKENS,
        ],
    }


__all__ = [
    "BOS",
    "EOS",
    "PAD",
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "TURN_END",
    "SEMANTIC_TOKENS",
    "RESERVED_TOKENS",
    "SPECIAL_TOKENS",
    "NUM_RESERVED",
    "special_tokens_map",
]
