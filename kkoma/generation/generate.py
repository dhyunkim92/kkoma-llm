"""Autoregressive generation with optional KV cache (spec section 20).

Greedy decoding with and without the cache produces identical tokens; RoPE
position offsets are taken from the cache length so cached decoding rotates new
tokens by their absolute position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from kkoma.generation.sampling import apply_repetition_penalty, sample_next_token
from kkoma.model.model import KkomaModel


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    seed: Optional[int] = None
    use_cache: bool = True


@torch.no_grad()
def generate(
    model: KkomaModel,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate continuations for a batch of prompts.

    ``input_ids``: ``[batch, prompt_len]``. Returns ``[batch, prompt_len + new]``.
    """

    # Explicit context-limit policy (spec 20.2), checked up front so both the
    # cached and uncached paths fail fast instead of erroring mid-generation
    # (cached), indexing past the learned position table, or silently
    # extrapolating RoPE beyond the trained context (uncached).
    total = input_ids.size(1) + config.max_new_tokens
    limit = model.config.context_length
    if total > limit:
        raise ValueError(
            f"prompt ({input_ids.size(1)}) + max_new_tokens "
            f"({config.max_new_tokens}) = {total} exceeds context_length "
            f"{limit}; shorten the prompt or reduce max_new_tokens"
        )

    model.eval()
    device = input_ids.device
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    generated = input_ids
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

    cache = model.new_cache() if config.use_cache else None

    # Prefill.
    logits, _ = model(input_ids, cache=cache)
    next_logits = logits[:, -1, :]

    for _ in range(config.max_new_tokens):
        next_logits = apply_repetition_penalty(
            next_logits, generated, config.repetition_penalty
        )
        next_token = sample_next_token(
            next_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            generator=generator,
        )
        # Once a sequence is finished, keep emitting eos (or its last token).
        if config.eos_token_id is not None:
            next_token = torch.where(
                finished, torch.full_like(next_token, config.eos_token_id), next_token
            )
        generated = torch.cat([generated, next_token[:, None]], dim=1)

        if config.eos_token_id is not None:
            finished = finished | (next_token == config.eos_token_id)
            if bool(finished.all()):
                break

        if cache is not None:
            logits, _ = model(next_token[:, None], cache=cache)
            next_logits = logits[:, -1, :]
        else:
            logits, _ = model(generated, cache=None)
            next_logits = logits[:, -1, :]

    return generated


__all__ = ["generate", "GenerationConfig"]
