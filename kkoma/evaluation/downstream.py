"""Zero/few-shot downstream evaluation primitives (spec section 21.3).

Provides likelihood-based scoring used by multiple-choice benchmarks
(HellaSwag, PIQA, ARC-Easy, WinoGrande) and a LAMBADA-style last-token
accuracy. Dataset loading lives in scripts; here we keep the scoring core so
the tokenizer and prompt format stay fixed.

The absolute scores of small models matter less than how they move with model
size and training stage, so these helpers focus on consistent measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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


def load_examples(path: str) -> list[MultipleChoiceExample]:
    """Read a frozen task file from scripts/prepare_downstream_data.py.

    Each choice already carries its leading space; the prompt format is fixed
    at preparation time so scoring stays free of per-task logic.
    """

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(
                MultipleChoiceExample(
                    context=row["context"],
                    choices=list(row["choices"]),
                    label=int(row["label"]),
                )
            )
    return out


@torch.no_grad()
def score_choices(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    examples: list[MultipleChoiceExample],
    device: torch.device,
    batch_size: int = 16,
) -> list[list[float]]:
    """Length-normalized log p(choice | context) for every choice.

    Batched: one forward covers ``batch_size`` (context, choice) pairs. The
    unbatched path costs one forward per pair, which for a 500-question
    4-way task is 2,000 near-empty forwards.

    Sequences are right-padded. That is sound only because attention here is
    causal and carries no key-padding mask: a real token at position i attends
    to positions <= i, so trailing pad is unreachable. Left padding would shift
    every RoPE position and corrupt the scores silently.
    """

    # Flatten to pairs, remembering where each one belongs.
    flat: list[tuple[int, int, list[int], list[int]]] = []
    for ei, ex in enumerate(examples):
        ctx_ids = tokenizer.encode(ex.context, add_bos=True)
        for ci, choice in enumerate(ex.choices):
            flat.append((ei, ci, ctx_ids, tokenizer.encode(choice)))

    scores: list[list[float]] = [[float("-inf")] * len(ex.choices) for ex in examples]
    # Group similar lengths together so padding waste (and the peak logits
    # tensor, which is batch x seq x 32,768) stays small.
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][2]) + len(flat[i][3]))
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0

    for start in range(0, len(order), batch_size):
        chunk = [flat[i] for i in order[start : start + batch_size]]
        chunk = [c for c in chunk if c[3]]  # empty continuation stays -inf
        if not chunk:
            continue

        widths = [len(c[2]) + len(c[3]) for c in chunk]
        width = max(widths)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        for r, (_, _, ctx, cont) in enumerate(chunk):
            ids[r, : len(ctx) + len(cont)] = torch.tensor(ctx + cont, device=device)

        logits, _ = model(ids)
        # Entry j of tok_lp scores ids[:, j+1]; gather the gold token and
        # subtract logsumexp rather than materializing a full log_softmax.
        lg = logits[:, :-1].float()
        tok_lp = lg.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)

        pos = torch.arange(width - 1, device=device).unsqueeze(0)
        c_len = torch.tensor([len(c[2]) for c in chunk], device=device).unsqueeze(1)
        t_len = torch.tensor(widths, device=device).unsqueeze(1)
        # Continuation token at index i is predicted by entry i-1.
        mask = (pos >= c_len - 1) & (pos < t_len - 1)
        summed = (tok_lp * mask).sum(dim=1)
        norm = summed / mask.sum(dim=1).clamp(min=1)

        for r, (ei, ci, _, _) in enumerate(chunk):
            scores[ei][ci] = float(norm[r])

    return scores


@torch.no_grad()
def evaluate_task(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    examples: list[MultipleChoiceExample],
    device: torch.device,
    batch_size: int = 16,
    reduce_fn=None,
) -> dict:
    """acc_norm plus two margins, optionally reduced across DDP ranks.

    Accuracy over 500 questions is noisy (about +/-2% at chance), and near
    chance it barely moves while the model is plainly still learning. The
    margins do move: they are continuous in the log-likelihoods rather than
    thresholded by an argmax.

    - ``margin_max``  gold minus the *strongest* distractor. Positive exactly
      when the answer is correct, so it reads as a signed confidence.
    - ``margin_mean`` gold minus the average distractor. Smoother, and rises
      before any answer flips.

    ``reduce_fn`` must sum a tensor across ranks. It is called exactly once,
    unconditionally: skipping it on a rank with no examples would desynchronize
    the collective and hang the run.
    """

    model.eval()
    scores = score_choices(model, tokenizer, examples, device, batch_size=batch_size)

    acc = torch.zeros(4, device=device)  # [correct, margin_max, margin_mean, n]
    for ex, row in zip(examples, scores):
        gold = row[ex.label]
        others = [s for i, s in enumerate(row) if i != ex.label]
        if not others:
            continue
        acc[0] += float(row.index(max(row)) == ex.label)
        acc[1] += gold - max(others)
        acc[2] += gold - (sum(others) / len(others))
        acc[3] += 1.0

    if reduce_fn is not None:
        acc = reduce_fn(acc)
    n = acc[3].item()
    if n == 0:
        return {"acc_norm": float("nan"), "margin_max": float("nan"),
                "margin_mean": float("nan"), "n": 0}
    return {
        "acc_norm": acc[0].item() / n,
        "margin_max": acc[1].item() / n,
        "margin_mean": acc[2].item() / n,
        "n": int(n),
    }


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
    "evaluate_task",
    "evaluate_lambada",
    "load_examples",
    "score_choices",
]
