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


# ---------------------------------------------------------------------------
# Tokenizer evaluation metrics (kkoma/tokenizer/evaluate.py).
#
# Untested before the 2026-07 audit, which found two of them carried no signal:
# `byte_token_fraction` was structurally always 0 (a ByteLevel pre-tokenizer
# makes byte_fallback unreachable) and `roundtrip_ok` only asserted the decode
# was non-empty. Both are written into artifacts/tokenizer/evaluation.json.
# ---------------------------------------------------------------------------


def test_roundtrip_ok_is_exact_equality_not_just_non_empty(tiny_tokenizer):
    """The old check was `decode(...).strip() != ""`, which a tokenizer that
    mangled every Korean character would still pass. Pin that the reported flag
    tracks real equality by exercising both outcomes on the same tokenizer."""

    from kkoma.tokenizer.evaluate import QUALITATIVE_SENTENCES

    for s in QUALITATIVE_SENTENCES:
        decoded = tiny_tokenizer.decode(tiny_tokenizer.encode(s), skip_special_tokens=False)
        assert (decoded == s) is True, f"fixture tokenizer lost text: {s!r}"

    # And the comparison is not vacuous: a decode of a *different* id sequence
    # is non-empty yet unequal, which the old check would have called OK.
    other = tiny_tokenizer.decode(tiny_tokenizer.encode("something else"),
                                  skip_special_tokens=False)
    assert other.strip() != ""
    assert other != QUALITATIVE_SENTENCES[0]


def test_single_char_fraction_reflects_fragmentation(tiny_tokenizer):
    """The metric it replaced was always 0; this one has to move with the text."""

    from kkoma.tokenizer.evaluate import evaluate_corpus

    # The fixture corpus is English, so Korean has no learned merges and
    # fragments into single characters far more often.
    en = evaluate_corpus(tiny_tokenizer, ["The model is trained from scratch."])
    ko = evaluate_corpus(tiny_tokenizer, ["이 모델은 처음부터 직접 학습되었습니다."])
    assert 0.0 <= en["single_char_token_fraction"] <= 1.0
    assert ko["single_char_token_fraction"] > en["single_char_token_fraction"]


def test_special_tokens_single_id_requires_exactly_one_id(tiny_tokenizer):
    """The reported flag used to be `token_to_id(t) is not None`, which stays
    True even if the pre-tokenizer splits the token into pieces."""

    from kkoma.tokenizer.special_tokens import SEMANTIC_TOKENS

    for t in SEMANTIC_TOKENS:
        assert tiny_tokenizer.encode(t, allow_special=True) == [tiny_tokenizer.token_to_id(t)]
