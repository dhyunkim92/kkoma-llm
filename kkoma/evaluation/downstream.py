"""Zero/few-shot downstream evaluation primitives (spec section 21.3).

Provides likelihood-based scoring used by multiple-choice benchmarks
(HellaSwag, PIQA, ARC-Easy, WinoGrande) and a LAMBADA-style last-token
accuracy. Dataset loading lives in scripts; here we keep the scoring core so
the tokenizer and prompt format stay fixed.

The absolute scores of small models matter less than how they move with model
size and training stage, so these helpers focus on consistent measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from kkoma.model.model import KkomaModel
from kkoma.tokenizer.utils import KkomaTokenizer


@torch.no_grad()
def continuation_logprob(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    context: str,
    continuation: str,
    device: torch.device,
) -> tuple[float, int]:
    """Sum log p(continuation | context) and the continuation token count."""

    ctx_ids = tokenizer.encode(context, add_bos=True)
    cont_ids = tokenizer.encode(continuation)
    if not cont_ids:
        return float("-inf"), 0

    input_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)[0]

    # The token at position i predicts the token at i+1.
    total = 0.0
    start = len(ctx_ids)
    for i in range(start, len(ctx_ids) + len(cont_ids)):
        target = input_ids[0, i]
        total += log_probs[i - 1, target].item()
    return total, len(cont_ids)


@dataclass
class MultipleChoiceExample:
    context: str
    choices: list[str]
    label: int


@torch.no_grad()
def evaluate_multiple_choice(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    examples: list[MultipleChoiceExample],
    device: torch.device,
    length_normalized: bool = True,
) -> dict:
    """Accuracy under per-choice continuation log-likelihood."""

    model.eval()
    correct = 0
    for ex in examples:
        scores = []
        for choice in ex.choices:
            lp, n = continuation_logprob(model, tokenizer, ex.context, choice, device)
            scores.append(lp / n if (length_normalized and n) else lp)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == ex.label)
    n = max(len(examples), 1)
    return {"accuracy": correct / n, "n": len(examples)}


@torch.no_grad()
def evaluate_lambada(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    examples: list[str],
    device: torch.device,
) -> dict:
    """Last-word prediction accuracy (greedy) over LAMBADA-style passages."""

    model.eval()
    correct = 0
    for passage in examples:
        words = passage.rsplit(" ", 1)
        if len(words) != 2:
            continue
        context, last_word = words
        lp, _ = continuation_logprob(model, tokenizer, context, " " + last_word, device)
        # Greedy check: does the model assign the gold last word max prob?
        ctx_ids = tokenizer.encode(context, add_bos=True)
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        pred_id = int(logits[0, -1].argmax().item())
        gold_id = tokenizer.encode(" " + last_word)[0]
        correct += int(pred_id == gold_id)
    n = max(len(examples), 1)
    return {"accuracy": correct / n, "n": len(examples)}


__all__ = [
    "MultipleChoiceExample",
    "continuation_logprob",
    "evaluate_multiple_choice",
    "evaluate_lambada",
]
