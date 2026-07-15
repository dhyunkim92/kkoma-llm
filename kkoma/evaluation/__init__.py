"""Evaluation: language modeling, generation, efficiency, downstream."""

from kkoma.evaluation.efficiency import benchmark_forward, estimate_flops_per_token
from kkoma.evaluation.generation import generate_samples
from kkoma.evaluation.language_modeling import evaluate_bilingual, evaluate_lm_loss

__all__ = [
    "evaluate_lm_loss",
    "evaluate_bilingual",
    "generate_samples",
    "benchmark_forward",
    "estimate_flops_per_token",
]
