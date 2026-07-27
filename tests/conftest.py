"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from kkoma.config import ModelConfig
from kkoma.model.model import KkomaModel


def tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        context_length=64,
        n_layer=2,
        d_model=64,
        n_head=4,
        n_kv_head=2,
        head_dim=16,
        norm="rmsnorm",
        positional_encoding="rope",
        activation="swiglu",
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def tiny_model() -> KkomaModel:
    torch.manual_seed(0)
    return KkomaModel(tiny_config())


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory):
    """Directory holding the trained fixture tokenizer.

    Separate from ``tiny_tokenizer`` because anything that goes through a
    ``RunConfig`` (``build_tokenizer``, ``run_training``) needs the path, not the
    object. Session-scoped so the BPE is still trained only once.
    """

    pytest.importorskip("tokenizers")
    from kkoma.tokenizer.train import TokenizerTrainConfig, train_tokenizer

    out = tmp_path_factory.mktemp("tok")
    corpus = out / "corpus.txt"
    text = (
        "The model is trained from scratch.\n"
        "이 모델은 처음부터 직접 학습되었습니다.\n"
        "Kkoma-LLM은 영어와 한국어를 모두 처리합니다.\n"
        "Artificial intelligence can do many things.\n"
    ) * 50
    corpus.write_text(text, encoding="utf-8")

    # 256 byte alphabet + 23 special tokens = 279; a small margin of merges
    # keeps the exact-size assertion reachable from a tiny corpus.
    cfg = TokenizerTrainConfig(
        vocab_size=300, min_frequency=1, output_dir=str(out), input_files=[str(corpus)]
    )
    train_tokenizer(cfg)
    return str(out)


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir):
    from kkoma.tokenizer.utils import KkomaTokenizer

    return KkomaTokenizer.from_file(tiny_tokenizer_dir)
