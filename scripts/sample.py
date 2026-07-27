"""Generate text from a checkpoint.

    python scripts/sample.py \
        --checkpoint artifacts/checkpoints/base_125m/final.pt \
        --config configs/pretraining/base_125m.yaml \
        --prompt "In the future, small language models" \
        --max-new-tokens 100 --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from kkoma.config import RunConfig
from kkoma.generation.generate import GenerationConfig, generate
from kkoma.model.model import KkomaModel
from kkoma.training.checkpoint import load_checkpoint
from scripts._common import build_tokenizer
from scripts._console import done, line, ok, section


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a Kkoma checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = section(f"loading checkpoint on {device}")
    model = KkomaModel(config.model).to(device)
    # map_location="cpu" so the checkpoint's optimizer state (unused here, ~2x
    # the weights) never reaches the GPU; load_state_dict copies into the
    # already-placed CUDA parameters. See scripts/evaluate.py for the numbers.
    load_checkpoint(args.checkpoint, model, map_location="cpu", restore_rng=False)
    model.eval()
    tokenizer = build_tokenizer(config)
    line(f"{config.project.run_name} | {model.num_parameters() / 1e6:.1f}M params | {args.checkpoint}")
    done(t0)

    ids = tokenizer.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_id,
        seed=args.seed,
        use_cache=True,
    )
    t0 = section(
        f"generating (max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p}, seed={args.seed})"
    )
    out = generate(model, input_ids, gen_cfg)[0].tolist()
    done(t0)

    # The generated text is the product, so print it plainly (unindented).
    print()
    print(tokenizer.decode(out))
    ok(f"{len(out) - len(ids)} tokens generated")


if __name__ == "__main__":
    main()
