"""Zero/few-shot downstream evaluation primitives (spec section 21.3).

Provides likelihood-based scoring used by multiple-choice benchmarks
(HellaSwag, PIQA, ARC, BoolQ, OpenBookQA, the KoBEST suite, and the
per-choice-context WinoGrande variant) plus a LAMBADA-style last-token
accuracy. Dataset loading lives in scripts; here we keep the scoring core so
the tokenizer and prompt format stay fixed.

The absolute scores of small models matter less than how they move with model
size and training stage, so these helpers focus on consistent measurement.
"""

from __future__ import annotations

import json
import math
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
    # Per-choice context, used only by WinoGrande-style tasks where each option
    # fills a blank and the *continuation* (the shared sentence suffix) is scored
    # under a different prefix per option. When None the single ``context`` is
    # shared across all choices, which is the ordinary multiple-choice case.
    contexts: list[str] | None = None


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
        contexts = ex.contexts if ex.contexts is not None else [ex.context] * len(ex.choices)
        scores = []
        for ctx, choice in zip(contexts, ex.choices):
            lp, n = continuation_logprob(model, tokenizer, ctx, choice, device)
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
                    context=row.get("context", ""),
                    choices=list(row["choices"]),
                    label=int(row["label"]),
                    contexts=row.get("contexts"),
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

    # Flatten to pairs, remembering where each one belongs. WinoGrande-style
    # tasks carry a per-choice context (``ex.contexts``); ordinary tasks share
    # one context, encoded once and reused across the choices.
    flat: list[tuple[int, int, list[int], list[int]]] = []
    for ei, ex in enumerate(examples):
        if ex.contexts is not None:
            ctx_ids_per_choice = [tokenizer.encode(c, add_bos=True) for c in ex.contexts]
        else:
            shared = tokenizer.encode(ex.context, add_bos=True)
            ctx_ids_per_choice = [shared] * len(ex.choices)
        for ci, choice in enumerate(ex.choices):
            flat.append((ei, ci, ctx_ids_per_choice[ci], tokenizer.encode(choice)))

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

    # [correct, margin_max, margin_mean, n, chance]. ``chance`` rides along in the
    # same tensor so the DDP reduce covers it too: choice counts vary per question
    # (ARC ships 3-, 4- and 5-way items), so a rank's shard has its own chance
    # level and only the summed version is meaningful.
    acc = torch.zeros(5, device=device)
    for ex, row in zip(examples, scores):
        gold = row[ex.label]
        others = [s for i, s in enumerate(row) if i != ex.label]
        if not others:
            continue
        acc[0] += float(row.index(max(row)) == ex.label)
        acc[1] += gold - max(others)
        acc[2] += gold - (sum(others) / len(others))
        acc[3] += 1.0
        acc[4] += 1.0 / len(row)

    if reduce_fn is not None:
        acc = reduce_fn(acc)
    n = acc[3].item()
    if n == 0:
        return {"acc_norm": float("nan"), "margin_max": float("nan"),
                "margin_mean": float("nan"), "n": 0,
                "chance": float("nan"), "above_chance": float("nan")}
    acc_norm = acc[0].item() / n
    chance = acc[4].item() / n
    return {
        "acc_norm": acc_norm,
        # Random-guess accuracy for this task. Reported because the suite mixes
        # 2-way and 4-way tasks, so raw accuracies are not comparable across
        # tasks and a plain average of them is anchored by the 2-way ones.
        "chance": chance,
        "above_chance": acc_norm - chance,
        "margin_max": acc[1].item() / n,
        "margin_mean": acc[2].item() / n,
        "n": int(n),
        # Binomial standard error at this n; 2*se is the rough noise floor on
        # any single-task comparison.
        "se": math.sqrt(max(acc_norm * (1.0 - acc_norm), 0.0) / n) if n else float("nan"),
    }


@torch.no_grad()
def evaluate_downstream_suite(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    tasks: list[tuple[str, str, list[MultipleChoiceExample]]],
    device: torch.device,
    batch_size: int = 16,
    reduce_fn=None,
    on_task=None,
) -> dict:
    """Score a list of ``(name, language, examples)`` tasks in one pass.

    Returns per-task ``acc_norm``/margins plus ``en_avg`` / ``ko_avg`` /
    ``overall_avg`` language aggregates — the shape the post-training evaluation
    (scripts/evaluate.py) writes out. The trainer keeps its own DDP-sharded loop
    because it also streams each task to W&B as it goes.

    ``on_task(name, language, result)`` is called after each task finishes, so a
    caller can print progress live rather than waiting for the whole suite.
    """

    per_task: dict = {}
    by_lang: dict[str, list[dict]] = {}
    for name, language, examples in tasks:
        r = evaluate_task(
            model, tokenizer, examples, device,
            batch_size=batch_size, reduce_fn=reduce_fn,
        )
        per_task[name] = {"language": language, **r}
        by_lang.setdefault(language, []).append(r)
        if on_task is not None:
            on_task(name, language, r)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _agg(rs: list[dict]) -> dict:
        """Raw mean plus the chance-corrected views.

        ``*_avg`` alone is not comparable across languages here: the EN suite is
        mostly 4-way (chance ~0.36) while the KO suite is mostly 2-way (chance
        ~0.45), so a raw average is anchored by how many binary tasks a language
        happens to contain. ``*_above_chance`` subtracts that floor and
        ``*_normalized`` rescales it to [0, 1] of the available headroom.
        """
        return {
            "avg": _mean([r["acc_norm"] for r in rs]),
            "chance": _mean([r["chance"] for r in rs]),
            "above_chance": _mean([r["above_chance"] for r in rs]),
            "normalized": _mean(
                [r["above_chance"] / (1.0 - r["chance"]) for r in rs if r["chance"] < 1.0]
            ),
            "n_tasks": len(rs),
        }

    aggregates: dict = {}
    for lang in ("en", "ko"):
        if lang in by_lang:
            for k, v in _agg(by_lang[lang]).items():
                aggregates[f"{lang}_{k}" if k != "avg" else f"{lang}_avg"] = v
    every = [r for rs in by_lang.values() for r in rs]
    for k, v in _agg(every).items():
        aggregates[f"overall_{k}" if k != "avg" else "overall_avg"] = v
    return {"tasks": per_task, "aggregates": aggregates}


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
    "evaluate_downstream_suite",
    "evaluate_lambada",
    "load_examples",
    "score_choices",
]
