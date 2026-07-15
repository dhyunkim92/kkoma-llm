"""Kkoma-LLM: a compact, reproducible decoder-only language-model project."""

from kkoma.config import (
    ModelConfig,
    RunConfig,
    TokenizerConfig,
)

__version__ = "0.1.0"

__all__ = ["ModelConfig", "RunConfig", "TokenizerConfig", "__version__"]
