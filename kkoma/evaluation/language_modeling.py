"""Language-modeling evaluation (spec section 21.1).

Token-weighted cross-entropy over a validation loader, reported as loss and
perplexity. English and Korean validation sets are evaluated separately and a
token-weighted aggregate is also produced.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from kkoma.model.model import KkomaModel


@torch.no_grad()
def evaluate_lm_loss(
    model: KkomaModel,
    loader: Iterable[dict],
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """Compute token-weighted loss and perplexity over a loader."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # Sum (not mean) so we can token-weight across batches.
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 50 else float("inf"),
        "tokens": total_tokens,
    }


def evaluate_bilingual(
    model: KkomaModel,
    loaders: dict,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """Evaluate per-language loaders and produce a token-weighted aggregate.

    ``loaders`` maps a language tag ("en"/"ko") to a validation loader.
    """

    results = {}
    agg_loss, agg_tokens = 0.0, 0
    for lang, loader in loaders.items():
        r = evaluate_lm_loss(model, loader, device, max_batches=max_batches)
        results[lang] = r
        agg_loss += r["loss"] * r["tokens"]
        agg_tokens += r["tokens"]
    mean = agg_loss / max(agg_tokens, 1)
    results["all"] = {
        "loss": mean,
        "perplexity": math.exp(mean) if mean < 50 else float("inf"),
        "tokens": agg_tokens,
    }
    return results


__all__ = ["evaluate_lm_loss", "evaluate_bilingual"]
