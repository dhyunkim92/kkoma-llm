"""Fixed-prompt generation samples (spec section 21.4).

The same English/Korean prompt set and sampling seed are used at every
checkpoint so qualitative drift across training is comparable.
"""

from __future__ import annotations

from typing import Optional

import torch

from kkoma.generation.generate import GenerationConfig, generate
from kkoma.model.model import KkomaModel
from kkoma.tokenizer.utils import KkomaTokenizer

DEFAULT_PROMPTS_EN = [
    "The meaning of life is",
    "Artificial intelligence can",
    "In the future, small language models",
]
DEFAULT_PROMPTS_KO = [
    "인공지능이란",
    "대한민국의 수도는",
    "작은 언어 모델을 직접 학습하면",
]


@torch.no_grad()
def generate_samples(
    model: KkomaModel,
    tokenizer: KkomaTokenizer,
    device: torch.device,
    prompts: Optional[list[str]] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 1234,
) -> list[dict]:
    """Generate one continuation per prompt and return prompt/completion pairs."""

    prompts = prompts if prompts is not None else DEFAULT_PROMPTS_EN + DEFAULT_PROMPTS_KO
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
        seed=seed,
        use_cache=True,
    )

    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, input_ids, gen_cfg)[0].tolist()
        completion = tokenizer.decode(out[len(ids):])
        samples.append(
            {"prompt": prompt, "completion": completion, "token_ids": out}
        )
    return samples


__all__ = ["generate_samples", "DEFAULT_PROMPTS_EN", "DEFAULT_PROMPTS_KO"]
