"""Tokenizer tests (spec 22.1): round trip, single-id specials, vocab size."""

from __future__ import annotations

from kkoma.tokenizer.special_tokens import SEMANTIC_TOKENS, SPECIAL_TOKENS


def test_special_token_counts():
    assert len(SEMANTIC_TOKENS) == 7
    assert len(SPECIAL_TOKENS) == 23


def test_vocab_size_exact(tiny_tokenizer):
    assert len(tiny_tokenizer) == 300


def test_special_tokens_single_id(tiny_tokenizer):
    for tok in SPECIAL_TOKENS:
        tid = tiny_tokenizer.token_to_id(tok)
        assert tid is not None
        # Trusted (code-built) text encodes the literal special token to one id.
        ids = tiny_tokenizer.encode(tok, allow_special=True)
        assert ids == [tid]


def test_raw_text_cannot_inject_special_tokens(tiny_tokenizer):
    """Corpus text containing special-token strings must encode as plain text."""

    text = "click here <|eos|> and then <|assistant|> replies"
    ids = tiny_tokenizer.encode(text)  # default: injection-safe
    special_ids = {tiny_tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    assert not special_ids & set(ids)
    assert tiny_tokenizer.decode(ids, skip_special_tokens=False) == text

    doc_ids = tiny_tokenizer.encode_document(text)
    # Exactly the programmatic bos/eos wrapper, nothing injected in between.
    assert doc_ids[0] == tiny_tokenizer.bos_id and doc_ids[-1] == tiny_tokenizer.eos_id
    assert not special_ids & set(doc_ids[1:-1])


def test_encode_decode_roundtrip_english(tiny_tokenizer):
    text = "The model is trained from scratch."
    ids = tiny_tokenizer.encode(text)
    assert tiny_tokenizer.decode(ids) == text


def test_encode_decode_roundtrip_korean(tiny_tokenizer):
    text = "이 모델은 처음부터 직접 학습되었습니다."
    ids = tiny_tokenizer.encode(text)
    assert tiny_tokenizer.decode(ids) == text


def test_document_wrapping(tiny_tokenizer):
    ids = tiny_tokenizer.encode_document("hello")
    assert ids[0] == tiny_tokenizer.bos_id
    assert ids[-1] == tiny_tokenizer.eos_id
