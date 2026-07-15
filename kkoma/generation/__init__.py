"""Autoregressive generation and sampling."""

from kkoma.generation.generate import GenerationConfig, generate
from kkoma.generation.sampling import sample_next_token

__all__ = ["generate", "GenerationConfig", "sample_next_token"]
