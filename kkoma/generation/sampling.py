"""Token sampling utilities (spec section 20.1).

Supports temperature, top-k, and optional top-p (nucleus) and repetition
penalty. A per-call ``torch.Generator`` makes sampling reproducible from a seed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style repetition penalty over already-generated tokens."""

    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for b in range(logits.size(0)):
        unique = torch.unique(generated[b])
        scores = logits[b, unique]
        logits[b, unique] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Remove tokens once the cumulative prob exceeds top_p, but keep the token
    # that first crosses the threshold: shift the mask right by one.
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False  # always keep the most probable token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to the original (unsorted) vocabulary order.
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token id per row from the final-position logits.

    ``logits``: ``[batch, vocab]``. Greedy decoding when ``temperature == 0``.
    """

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    # Sample in fp32: fp16 logits overflow to inf at low temperature
    # (softmax -> NaN -> multinomial error) and lose tail probability mass.
    logits = logits.float() / temperature
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


__all__ = [
    "sample_next_token",
    "filter_top_k",
    "filter_top_p",
    "apply_repetition_penalty",
]
