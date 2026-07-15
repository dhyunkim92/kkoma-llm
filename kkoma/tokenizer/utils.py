"""Thin wrapper around a trained Hugging Face ``tokenizers.Tokenizer``.

Provides the encode/decode interface the rest of the codebase relies on,
including BOS/EOS document wrapping used by the pretraining packer.
"""

from __future__ import annotations

import os
from typing import Optional

from kkoma.tokenizer.special_tokens import BOS, EOS, PAD


class KkomaTokenizer:
    def __init__(self, tokenizer):  # tokenizer: tokenizers.Tokenizer
        self._tok = tokenizer
        self.bos_id = self._tok.token_to_id(BOS)
        self.eos_id = self._tok.token_to_id(EOS)
        self.pad_id = self._tok.token_to_id(PAD)
        if self.bos_id is None or self.eos_id is None:
            raise ValueError("tokenizer is missing required special tokens")

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "KkomaTokenizer":
        from tokenizers import Tokenizer

        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        return cls(Tokenizer.from_file(path))

    # ---- size -------------------------------------------------------------
    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return len(self)

    # ---- encode / decode --------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
    ) -> list[int]:
        """Encode text; bos/eos are appended by id when requested.

        By default a document that literally contains "<|eos|>" or
        "<|assistant|>" encodes those as ordinary text: matching them would
        fabricate document boundaries and role tokens during pretraining
        (spec 4.5 keeps role semantics untrained). ``allow_special=True``
        opts back in for trusted, code-built strings (e.g. the Kkoma-Chat
        template renderer).
        """

        # tokenizers naming is inverted: encode_special_tokens=True means
        # special-token strings are encoded AS ordinary text (not matched).
        self._tok.encode_special_tokens = not allow_special
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_document(self, text: str) -> list[int]:
        """Wrap a pretraining document as ``<|bos|> ... <|eos|>``."""

        return self.encode(text, add_bos=True, add_eos=True)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._tok.id_to_token(idx)


__all__ = ["KkomaTokenizer"]
