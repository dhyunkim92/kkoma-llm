<p align="center">
  <img src="kkoma-mascot-pixel.svg" width="260" alt="Kkoma resting in its nest"/>
</p>

<h1 align="center">Kkoma-LLM</h1>

<p align="center">English | <b><a href="README.ko.md">한국어</a></b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.0+"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

> **Kkoma-LLM is a small language model built from scratch to be understood, not just used.**
> *Kkoma* (꼬마) is the Korean word for a little kid: small and not yet fully grown, but built and
> raised by hand. That is exactly what this project is. Instead of shrinking a large commercial model
> or fine-tuning someone else's weights, Kkoma walks the **entire path of building an LLM**, from
> tokenizer to architecture to data to training, all the way to completion within limited compute.
> The goal is not SOTA. It is to design every piece, explain why it works, and reproduce it.

---

## Table of contents

1. [The Kkoma project at a glance](#1-the-kkoma-project-at-a-glance)
2. [Core design principles](#2-core-design-principles)
3. [Installation](#3-installation)
4. [Repository layout](#4-repository-layout)
5. [Quickstart](#5-quickstart)
6. [Phase-by-phase guide](#6-phase-by-phase-guide)
   - [Phase 0 · Tokenizer](#phase-0--tokenizer-preparation)
   - [Phase 1 · Architecture Study](#phase-1--architecture-study)
   - [Phase 2 · Base Pretraining](#phase-2--base-pretraining)
   - [Phase 3 · Korean Continued Pretraining](#phase-3--korean-continued-pretraining)
7. [Evaluation and generation](#7-evaluation-and-generation)
8. [Working with configs](#8-working-with-configs)
9. [Data directory conventions](#9-data-directory-conventions)
10. [Distributed training · checkpoints · reproducibility](#10-distributed-training--checkpoints--reproducibility)
11. [Tests](#11-tests)
12. [FAQ](#12-faq)
13. [Out of scope for v1 / what comes next](#13-out-of-scope-for-v1--what-comes-next)

---

## 1. The Kkoma project at a glance

Kkoma is not a single model but a family of projects that grow in capability step by step.

```
Kkoma-LLM   →   Kkoma-Chat   →   Kkoma-Agent
(this repo)     (SFT + DPO)      (tool use, agent)
```

**This repository (Kkoma-LLM)** covers only the tokenizer and pretraining line, in four phases.

| Phase | Stage | One-liner | Key artifact |
|---|---|---|---|
| Phase 0 | Tokenizer Preparation | byte-level BPE, vocab 32,768 | `artifacts/tokenizer/` |
| Phase 1 | Architecture Study | baseline vs modern, one component changed at a time | ablation results |
| Phase 2 | Base Pretraining | 125M / 350M / 800M / 1B / 1.3B trained from scratch | `Kkoma-*-Base` |
| Phase 3 | Korean Continued Pretraining | adapt the English-centric Base to Korean | `Kkoma-*-Ko-Base` |

SFT/DPO (Kkoma-Chat), tool use and agents (Kkoma-Agent), and RL-based post-training are
**out of scope for this repository**.

---

## 2. Core design principles

- **Separate what to build from what to reuse.** Model structure and the training loop, where the
  learning value lives, are written by hand (`kkoma/model`, `kkoma/training`). Low-level machinery
  such as the BPE algorithm, CUDA kernels, and distributed communication comes from proven
  libraries (`tokenizers`, `torch`). → **`transformers` is deliberately not used.**
- **Simple but modern.** The code stays at nanoGPT-level readability while modern components such
  as RoPE, RMSNorm, SwiGLU, GQA, and weight tying are swapped in and out through config.
- **Controlled comparison.** Architecture experiments change exactly one component at a time and
  keep the tokenizer, corpus, token budget, optimizer, and seeds fixed.
- **Reproducibility.** Configs, seeds, data manifests (with shard checksums), and parameter reports
  are stored alongside checkpoints.

---

## 3. Installation

```bash
# 1) virtual environment (e.g. venv)
python -m venv .venv && source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt
# or install as a package (recommended): import kkoma from anywhere
pip install -e .
```

The primary hardware target is **8× NVIDIA V100 (FP16, NCCL)**. Single-GPU and CPU also work
(use `precision: fp32` in the config for CPU smoke tests).

> **Training on GPU requires a CUDA build of torch.** If the default install is a CPU build,
> replace it:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

Required packages: `torch`, `tokenizers` (BPE backend), `datasets` (FineWeb streaming), `numpy`,
`pyyaml`, `tqdm`. `wandb` (logging) and `pytest` (tests) are optional.

---

## 4. Repository layout

```text
kkoma-llm/
├── kkoma/                     # library (hand-written)
│   ├── config.py              #   nested-dataclass RunConfig (YAML/JSON load & save)
│   ├── tokenizer/             #   BPE training, evaluation, runtime wrapper, special tokens
│   ├── model/                 #   RoPE/RMSNorm/SwiGLU/GQA, KV cache, the model itself
│   ├── data/                  #   cleaning, streaming, weighted mixture, token packing
│   ├── training/              #   optimizer, scheduler, DDP, checkpoints, training loop
│   ├── generation/            #   sampling + KV-cache autoregressive generation
│   └── evaluation/            #   LM loss (EN/KO), efficiency, downstream, generation samples
├── configs/                   # run configs (YAML)
│   ├── tokenizer/             #   tokenizer_32k.yaml
│   ├── architecture/          #   core_125m_* / core_350m_* (cumulative progressive, 5 each)
│   ├── pretraining/           #   base_{125m,350m,800m,1b,1.3b}.yaml
│   └── continued_pretraining/ #   ko_{125m,350m,800m,1b,1.3b}.yaml
├── scripts/                   # CLI: prepare_* / train_* / evaluate / sample
├── tests/                     # unit & integration tests (pytest)
└── artifacts/                 # tokenizer, checkpoints, logs, evaluation results
```

> `scripts/` is an execution folder, not part of the installed library.
> **Always run from the repository root.** Each script bootstraps the root onto the path, so both
> `python scripts/x.py` and `python -m scripts.x` work.

---

## 5. Quickstart

```bash
# Phase 0: tokenizer (mixture ratios, vocab etc. in configs/tokenizer/tokenizer_32k.yaml)
python scripts/prepare_tokenizer_data.py --config configs/tokenizer/tokenizer_32k.yaml \
    --output-dir data/tokenizer
python scripts/train_tokenizer.py --input "data/tokenizer/*.jsonl" \
    --config configs/tokenizer/tokenizer_32k.yaml

# Phase 1: architecture study (one component changed at a time)
python scripts/prepare_architecture_data.py --output-dir data/architecture --tokenizer artifacts/tokenizer
python scripts/train_architecture.py --config configs/architecture/core_125m_modern.yaml

# Phase 2: base pretraining (single GPU)
python scripts/prepare_pretraining_data.py --output-dir data/pretrain --tokenizer artifacts/tokenizer
# optional: frozen downstream benchmark sets, scored during training (see §7)
python scripts/prepare_downstream_data.py --output-dir data/downstream
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml

# Phase 2: multi-GPU (8 GPUs on one node via torchrun)
torchrun --nproc_per_node=8 scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml

# Phase 2: resume from a checkpoint (works single-GPU or multi-GPU; just add --resume)
torchrun --nproc_per_node=8 scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml \
    --resume artifacts/checkpoints/base-1b/step_00010000.pt

# Phase 3: Korean continued pretraining
python scripts/prepare_korean_data.py --output-dir data/korean --tokenizer artifacts/tokenizer
python scripts/train_continued.py --config configs/continued_pretraining/ko_1b.yaml \
    --init-from artifacts/checkpoints/base-1b/final.pt

# evaluation / generation
python scripts/evaluate.py --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml --output artifacts/evaluation/base_1b.json
python scripts/sample.py --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml --prompt "In the future, small language models"
```

Any training run can be continued with `--resume <checkpoint.pt>` (restores the model, optimizer,
scheduler, grad scaler, step counters, and RNG state).

> **First-run tip:** the real FineWeb data is very large. To verify the pipeline end to end first,
> do one pass with small values such as `--gb 1` (tokenizer) or `--train-tokens 5_000_000`
> (architecture).

---

## 6. Phase-by-phase guide

### Phase 0 · Tokenizer (Preparation)

**Goal**: build a dedicated tokenizer that handles both English and Korean with reasonable
efficiency and is **shared unchanged** by the later Chat and Agent stages.

**Spec** (spec §4): byte-level BPE, final vocab of **exactly 32,768**, no UNK (every byte
representable), 23 special tokens (7 semantic + 16 reserved).

- **Document boundaries, padding**: `<|bos|>` `<|eos|>` `<|pad|>`
- **Reserved for Chat** (in the vocab only here, no role semantics trained): `<|system|>` `<|user|>` `<|assistant|>` `<|turn_end|>`
- **Reserved for future (Agent) extensions**: `<|reserved_0|>` … `<|reserved_15|>`

**1) Sample the training data**: English FineWeb-Edu 75% + Korean FineWeb2 25%, ~20GB.

```bash
python scripts/prepare_tokenizer_data.py --config configs/tokenizer/tokenizer_32k.yaml \
    --output-dir data/tokenizer
```

| Option | Default | Description |
|---|---|---|
| `--config` | (none) | tokenizer YAML with mixture ratios and target GB |
| `--output-dir` | `data/tokenizer` | output folder for JSONL shards |
| `--gb` | sum of the config's `target_gb` (else 20.0) | target size after cleaning (bytes) |
| `--seed` | `42` | sampling seed |
| `--no-progress` | (off) | disable progress bars |

> Progress is shown at two levels: an **overall bar (target GB, ETA, shard count)** plus a
> **current-shard bar (documents)**.

**2) Train the tokenizer**

```bash
python scripts/train_tokenizer.py --input "data/tokenizer/*.jsonl" \
    --config configs/tokenizer/tokenizer_32k.yaml
```

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | glob of training text (`*.jsonl` or `*.txt`) |
| `--config` | (none) | tokenizer YAML with vocab, min_frequency, output dir |
| `--output-dir` | `artifacts/tokenizer` | tokenizer artifact folder (overrides config) |
| `--vocab-size` | `32768` | final size including special tokens (overrides config) |
| `--min-frequency` | `2` | minimum merge frequency |

**Artifacts**: `artifacts/tokenizer/` gets `tokenizer.json`, `vocab.json`, `merges.txt`,
`special_tokens_map.json`, `training_config.json`, `evaluation.json`, and more.

**Try it from Python**

```python
from kkoma.tokenizer.utils import KkomaTokenizer
tok = KkomaTokenizer.from_file("artifacts/tokenizer")
print(len(tok))                              # 32768
ids = tok.encode("작은 언어 모델을 직접 학습하면", add_bos=True)
print(ids, tok.decode(ids))
```

**Done when**: vocab is 32,768, every special token encodes to a single id, EN/KO round trips
are clean, Korean token efficiency improves over GPT-2, and the manifest is saved.

---

### Phase 1 · Architecture Study

**Goal**: implement the modern decoder-only components by hand and, under **identical
conditions**, stack them one at a time to decide the final structure for the Base models
(spec §6). The study runs at two scales: 125M and 350M.

**What is compared**: starting from the baseline (learned positions / LayerNorm / GELU / MHA),
RoPE → RMSNorm → SwiGLU → GQA are **accumulated** one by one up to modern, as a progressive set
at both scales (`core_125m_*`, `core_350m_*`). Each step changes exactly one component relative
to the previous step, so that component's effect is exposed directly, and repeating the same
comparison at two sizes shows whether the conclusions are robust to scale. Bold cells are the
components switched on relative to baseline. All configs share the tokenizer, corpus, token
budget, optimizer, and seeds; only the `model` section differs (spec §5.3, controlled comparison).

| config (`core_125m_*` / `core_350m_*`) | positions | norm | FFN | attention |
|---|---|---|---|---|
| `…_baseline` | learned | LayerNorm | GELU | MHA |
| `…_rope` | **RoPE** | LayerNorm | GELU | MHA |
| `…_rope_rmsnorm` | **RoPE** | **RMSNorm** | GELU | MHA |
| `…_rope_rmsnorm_swiglu` | **RoPE** | **RMSNorm** | **SwiGLU** | MHA |
| `…_modern` | **RoPE** | **RMSNorm** | **SwiGLU** | **GQA** |

| Scale | n_layer | d_model | n_head | modern n_kv_head | micro_batch | total params (≈) |
|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 | 3 | 8 | 99.5M – 111M |
| 350M | 24 | 1024 | 16 | 4 | 4 | 304M – 342M |

> It is expected that baseline (MHA, GELU) has more parameters than modern (GQA, SwiGLU): MHA's
> KV projections and the large GELU MLP intermediate account for it. Modern is identical in
> structure and size to the corresponding Base model.

**1) Prepare the fixed corpus**: FineWeb-Edu 95% + FineWeb2 Korean 5%, default 300M train /
10M validation tokens.

```bash
python scripts/prepare_architecture_data.py \
    --output-dir data/architecture --tokenizer artifacts/tokenizer \
    --train-tokens 300000000 --val-tokens 10000000
```

Data is written to per-language directories, re-mixed 95:5 at load time, and validation loss is
computed separately for EN/KO (→ [§9 Data directory conventions](#9-data-directory-conventions)).

**2) Train each variant** (only the config changes)

```bash
# single run example: 125M baseline
python scripts/train_architecture.py --config configs/architecture/core_125m_baseline.yaml

# run each scale's progressive set in sequence (5 configs each)
for cfg in configs/architecture/core_125m_*.yaml; do
    python scripts/train_architecture.py --config "$cfg"
done
for cfg in configs/architecture/core_350m_*.yaml; do
    python scripts/train_architecture.py --config "$cfg"
done
```

**Comparison metrics**: train/val loss, perplexity, gradient norm, loss spikes (quality);
tokens/s, peak memory, step time, parameter count, estimated FLOPs (efficiency); for GQA also
KV-cache size, generation latency, and generation tokens/s.

> **SwiGLU parameter fairness:** SwiGLU has two input projections, so to match block parameters
> with GELU its inner dimension is set to `~8/3 × d_model`, rounded to a multiple of 128
> (`kkoma/config.py:default_ffn_dim`).

---

### Phase 2 · Base Pretraining

**Goal**: train Base models of several sizes from scratch with the decided structure and analyze
scaling (spec §13–18).

> **The Base corpus is not English-only.** It is a mixture of **English FineWeb-Edu 95% + Korean
> FineWeb2 5%** (spec §14.1). "English-centric Base" means English dominates, not that Korean is
> absent: the 5% keeps the Korean token embeddings warm so Phase 3 does not start from scratch.
> `prepare_pretraining_data.py` therefore downloads **both** languages (Korean only the 5% share).

| Model | n_layer | d_model | n_head | n_kv_head | token budget | peak LR | config |
|---|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 | 3 | 1.25B | 6e-4 | `base_125m.yaml` |
| 350M | 24 | 1024 | 16 | 4 | 3.5B | 3e-4 | `base_350m.yaml` |
| 800M | 24 | 1536 | 24 | 6 | 8B | 2e-4 | `base_800m.yaml` |
| 1B | 24 | 1792 | 28 | 7 | 9B | 1.8e-4 | `base_1b.yaml` |
| 1.3B | 24 | 2048 | 32 | 8 | 11.5B | 1.5e-4 | `base_1.3b.yaml` |

> The names are approximations. The **actual parameter count** including the 32K vocab and weight
> tying is saved via `parameter_report` at training start (e.g. base_125m → ~99.5M, base_800m →
> ~645M, base_1b → ~879M, base_1.3b → ~1130M).
>
> **All of these fit a 32GB V100** (DDP replicates the model per GPU). Measured peak per rank
> (seq 1024, fp16 + AdamW): base_1b `micro_batch_size: 2` → 18.4 GB, base_1.3b `micro_batch_size: 2`
> → 22.6 GB (both are the config defaults). base_1.3b is near the ceiling: ~1.5B is the practical
> limit before activation checkpointing is needed.

**1) Prepare the corpus**

```bash
python scripts/prepare_pretraining_data.py \
    --output-dir data/pretrain --tokenizer artifacts/tokenizer --tokens 11.5e9
```

This streams **both** English (95%) and Korean (5%) into separate per-language directories:
`train/` + `val/` (English) and `train_ko/` + `val_ko/` (Korean). Only the needed share of each is
downloaded. With `--tokens 11.5e9`, Korean is ~0.58B tokens (5%), not all of FineWeb2. Build the
11.5B corpus once: 125M reads the first 1.25B tokens, 350M the first 3.5B, 800M the first 8B, 1B the
first 9B, and 1.3B all of it (how much is actually read is set by each config's `training.max_tokens`).

**1′) Prepare downstream benchmark sets (optional)**

Only needed if you want HellaSwag / ARC-Easy / KoBEST-HellaSwag scored during training (see §7).
Training runs fine without it and simply skips them.

```bash
python scripts/prepare_downstream_data.py --output-dir data/downstream
```

This samples 500 questions per task and freezes them to JSONL, then deletes the source downloads.
The sampling is fixed by `--seed` (default 42) and the pinned dataset revisions, so rebuilding
yields the identical questions and every model sees the same test.

| Task | Split | Language | Sampling |
|---|---|---|---|
| HellaSwag | validation | English | 500, stratified 125 per gold label |
| ARC-Easy | test | English | 500, random, original choice sets kept |
| KoBEST-HellaSwag | test | Korean | all 500 |

**2) Train (single GPU)**

```bash
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml
```

**2′) Train (multi-GPU, 8× V100)**

```bash
torchrun --nproc_per_node=8 scripts/train_pretraining.py \
    --config configs/pretraining/base_1b.yaml
```

**Batch-token arithmetic** (spec §15.4): the global batch targets roughly 256K tokens.

```
global_batch_tokens = micro_batch_size × sequence_length × grad_accum_steps × world_size
e.g. 4 × 1024 × 8 × 8 = 262,144
```

Leave `grad_accum_steps` unset and it is derived from `global_batch_tokens` automatically.

**Resuming after an interruption**

```bash
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml \
    --resume artifacts/checkpoints/base-1b/step_00010000.pt
```

Checkpoints are saved as `step_00010000.pt` (periodic), `best_val.pt` (lowest validation loss),
and `final.pt` (end of run). Every checkpoint also records reproducibility identifiers
(git commit, tokenizer sha256, data-manifest sha256, W&B run id) per spec §18.1 and §25.

On resume the W&B run id is read from the checkpoint so logging **continues in the same run**
(spec §23), and the data stream is **fast-forwarded** by the number of consumed blocks so
training continues with exactly the next batch it would have seen without the interruption
(spec §18.3). The fast-forward re-tokenizes the skipped prefix once, which can take a while for
large corpora; setting `training.resume_fastforward: false` restarts the stream from the
beginning instead, and that fact is recorded in the checkpoints.

**W&B projects** are split by training stage so runs stay organized. Every size in a stage shares
one project, and size/variant is carried by the run name (`base-1b-tpp10`, `ko-1b-cpt-2b`, …).
`project.name` stays `kkoma-llm`; only `logging.project` differs:

| Stage | Configs | W&B project |
|---|---|---|
| Base pretraining | `configs/pretraining/base_*.yaml` | `kkoma-llm-pt` |
| Continued pretraining | `configs/continued_pretraining/ko_*.yaml` | `kkoma-llm-cpt` |
| Architecture ablation | `configs/architecture/core_*.yaml` | `kkoma-llm-core` |

---

### Phase 3 · Korean Continued Pretraining

**Goal**: adapt an English-centric Base (English 95% / Korean 5% during pretraining) to a
Korean-heavy mixture while preserving English ability and measuring forgetting (spec §19). Input:
`Kkoma-<size>-Base` → output: `Kkoma-<size>-Ko-Base`.

**Defaults**: Korean 70% + English replay 30%, 2B tokens (≈1.4B KO / 0.6B EN), peak LR 5e-5.
The same recipe applies to every Base size. `ko_125m` / `ko_350m` / `ko_800m` / `ko_1b` /
`ko_1.3b` differ only in the model dims and match their `base_*` counterpart; the example below
uses 1B.
**The v1 default starts with a fresh optimizer state** (i.e. use only `--init-from`, not
`--resume`).

**1) Prepare the corpus**

```bash
python scripts/prepare_korean_data.py \
    --output-dir data/korean --tokenizer artifacts/tokenizer --tokens 2e9
```

**2) Start from the Base weights and train on Korean**

```bash
python scripts/train_continued.py \
    --config configs/continued_pretraining/ko_1b.yaml \
    --init-from artifacts/checkpoints/base-1b/final.pt
```

| Option | Description |
|---|---|
| `--init-from` | (required) Base checkpoint to adapt. Loads **weights only** (fresh optimizer) |
| `--resume` | resume an interrupted CPT run itself (full state restore) |

**Measuring forgetting**: the validation loaders are split into `en`/`ko`, so English and Korean
losses are logged separately during training. Catastrophic forgetting is read off how much
`val/loss_en` rises relative to the pre-adaptation Base.

**Downstream benchmarks apply here too** (§7). The CPT configs enable the same three tasks, and this
is where they matter most: the Base sees only 5% Korean, so `downstream/ko_avg` (KoBEST-HellaSwag)
sits near chance until Korean adaptation moves it. Prepare the sets first (Phase 2 step 1′) if you
have not already; the same `data/downstream/` files are reused, so nothing extra is needed when you
ran it for the Base. If they are missing, CPT logs a warning and skips them rather than failing.

---

## 7. Evaluation and generation

**Evaluate a checkpoint**: LM loss (EN/KO split), an efficiency benchmark, and fixed-prompt
generation samples, saved as JSON.

```bash
python scripts/evaluate.py \
    --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml \
    --output artifacts/evaluation/base_1b.json
```

| Option | Default | Description |
|---|---|---|
| `--max-batches` | `50` | number of validation batches |
| `--no-generation` | (off) | skip the generation-sample step |

**Downstream benchmarks during training**: if you prepared the frozen sets (Phase 2 step 1′,
optional), three zero-shot multiple-choice sets (HellaSwag, ARC-Easy, KoBEST-HellaSwag) are scored
on a separate cadence (`downstream_interval`, default every 1,000 steps, plus a step-0 baseline),
alongside the validation-loss curve.

Scoring is length-normalized continuation log-likelihood (`acc_norm`). Each task also logs a
`margin_max` and `margin_mean` (gold minus the strongest / average distractor): near chance the
accuracy barely moves while the margins still do, so they show the model is learning before any
answer flips. Aggregates are `downstream/en_avg`, `downstream/ko_avg`, and `downstream/overall_avg`.
At this scale (tpp ~10) HellaSwag stays near chance (25%) for a long time, and KoBEST barely moves
until Phase 3, so these are for watching the trend, not the absolute score. A 500-question set is
too noisy to select on, so **best-checkpoint selection stays on validation loss**. Turn the whole
thing off with `downstream_enabled: false`, or drop a single task with its `enabled: false`.

**Generate text**: KV-cache autoregressive generation. The same seed reproduces the same output.

```bash
python scripts/sample.py \
    --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml \
    --prompt "대한민국의 수도는" \
    --max-new-tokens 100 --temperature 0.8 --top-k 50
```

| Option | Default | Description |
|---|---|---|
| `--max-new-tokens` | `100` | number of tokens to generate |
| `--temperature` | `0.8` | 0 means greedy |
| `--top-k` / `--top-p` | `50` / `1.0` | sampling filters |
| `--seed` | `1234` | seed for reproducibility |

**Fixed evaluation prompts** (compared across all checkpoints with the same seed):
English `"The meaning of life is"`, `"Artificial intelligence can"`,
`"In the future, small language models"` / Korean `"인공지능이란"`, `"대한민국의 수도는"`,
`"작은 언어 모델을 직접 학습하면"`.

---

## 8. Working with configs

Every run is defined by a single YAML (`RunConfig`). The heart of it is the `model` section:
change one line here and you get a different architecture.

```yaml
model:
  vocab_size: 32768
  context_length: 1024
  n_layer: 12
  d_model: 768
  n_head: 12
  n_kv_head: 3              # equal to n_head -> MHA, smaller -> GQA
  norm: rmsnorm            # rmsnorm | layernorm
  positional_encoding: rope # rope | learned
  activation: swiglu       # swiglu | gelu
  tie_word_embeddings: true
  bias: false
  dropout: 0.0

training:
  precision: fp16          # use fp32 for CPU tests
  global_batch_tokens: 262144
  micro_batch_size: 4
  max_tokens: 1250000000
  eval_interval: 1000       # validation-loss cadence
  downstream_interval: 1000 # benchmark cadence (0 disables)

optimizer: { name: adamw, learning_rate: 0.0006, beta1: 0.9, beta2: 0.95, weight_decay: 0.1 }
scheduler: { name: cosine, warmup_ratio: 0.02, min_lr_ratio: 0.1 }

evaluation:                 # frozen sets from prepare_downstream_data.py (see §7)
  downstream_enabled: true
  downstream_tasks:
  - { name: hellaswag,        path: data/downstream/hellaswag.jsonl,        language: en }
  - { name: arc_easy,         path: data/downstream/arc_easy.jsonl,         language: en }
  - { name: kobest_hellaswag, path: data/downstream/kobest_hellaswag.jsonl, language: ko }
```

Build or inspect one from code:

```python
from kkoma.config import RunConfig
cfg = RunConfig.from_yaml("configs/pretraining/base_1b.yaml")

from kkoma.model.model import KkomaModel
m = KkomaModel(cfg.model)
print(m.parameter_report()["total"])        # actual parameter count
```

---

## 9. Data directory conventions

The `prepare_*` scripts write data into **per-language directories**. This (1) lets the config
weights control the mixture ratio at training time and (2) makes it possible to compute **EN/KO
validation loss separately** (spec §14.4).

```text
data/pretrain/
├── train/        # English (95% of the train budget)   → weight 0.95 in the config
├── train_ko/     # Korean  (5% of the train budget)    → weight 0.05 in the config
├── val/          # English validation                  → val/loss_en
└── val_ko/       # Korean validation                   → val/loss_ko
```

The training corpus is split by ratio at prepare time, and **`MixtureStream` interleaves it by
seed at load time**. Korean CPT uses the `train_ko` (70%) / `train_en` (30%) / `val_ko` /
`val_en` layout.

**Train/val separation is a document-level hash holdout** (spec §14.3). A document goes to the
validation pool iff `sha256(text) % holdout_mod == 0` (1% by default), so train and val cannot
overlap regardless of stream order or seeds. Document order is seed-shuffled at prepare time and
then fixed (spec §14.2, `--shuffle-buffer`, default 10,000).

`data_manifest.json` records the sources, seeds, shuffle settings, holdout method, document and
token counts, and per-shard SHA-256 checksums.

**Downstream evaluation sets** live separately under `data/downstream/` (spec §14.5), written by
`prepare_downstream_data.py` rather than the `prepare_*` corpus scripts. Each is a small frozen
JSONL of 500 questions, with `downstream_manifest.json` recording the source split, original size,
sample count, seed, dataset revision, and file checksum.

```text
data/downstream/
├── hellaswag.jsonl           # 500, stratified by gold label
├── arc_easy.jsonl            # 500, original choice sets kept
├── kobest_hellaswag.jsonl    # 500 (whole test split)
└── downstream_manifest.json
```

---

## 10. Distributed training · checkpoints · reproducibility

- **DDP**: one process per GPU, NCCL backend. Launch with `torchrun --nproc_per_node=N` and the
  code configures itself from the environment (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`). During gradient
  accumulation, `no_sync()` avoids unnecessary all-reduces and synchronization happens only on
  the last micro-step.
- **Mixed precision (FP16)**: autocast + GradScaler. Every log interval checks loss finiteness,
  grad norm, scaler scale, and skipped steps; NaN/Inf steps are skipped (in lockstep across
  ranks).
- **Checkpoints**: save the model, optimizer, scheduler, scaler, global step, tokens processed,
  RNG state, config, and provenance identifiers. Weight tying is restored to the same Parameter
  object after loading. Only the last N periodic checkpoints are kept, with best kept separately.
- **Reproducibility**: global/data/init/sampling seeds are managed separately and stored with
  the config. Data manifests, parameter reports, and seeds are all recorded, so identical
  conditions can be reproduced.

---

## 11. Tests

```bash
pytest                 # everything
pytest tests/test_kv_cache.py -v   # a subset
```

Coverage: tokenizer round trip / single-id special tokens / vocab size, RoPE shape, offset, and
determinism, GQA head mapping, causal masking, RMSNorm numerics, SwiGLU shapes, **weight tying
(survives save/load)**, **KV-cache equivalence (cache on/off greedy outputs match)**, checkpoint
resume, determinism, small overfit, and the scheduler's warmup/decay shape.

---

## 12. FAQ

**`ModuleNotFoundError: No module named 'kkoma'`**
You are not running from the repository root. Run `python scripts/x.py` from the root, or
`pip install -e .` first.

**Why not `transformers`?**
Deliberate. Implementing the model, generation, and training loop by hand is the point of the
project. From the HF ecosystem only `tokenizers` (BPE backend) and `datasets` (streaming) are
used.

**Is all the data downloaded up front?**
No. `datasets` is used with `streaming=True`, so only as much as is consumed comes over the
network. The progress bars track **what is written to local shards**, so they reflect real
progress.

**Can I run it on CPU?**
For pipeline checks, yes. Set `precision: fp32` in the config and use a small token budget.
Real training needs a GPU (e.g. V100) and a CUDA build of torch.

**Do I have to build the tokenizer first?**
Yes. Training and evaluation configs reference `tokenizer.path: artifacts/tokenizer`, so Phase 0
must come first.

**What is `kkoma_llm.egg-info/`?**
Install metadata created by `pip install -e .`. It is gitignored and regenerates if deleted.

---

## 13. Out of scope for v1 / what comes next

**Excluded** from this repository (Kkoma-LLM v1): SFT and DPO, PPO/GRPO/RLHF, tool use, function
calling, agents, long-context pretraining, Mixture-of-Experts, and large-scale synthetic data.

The follow-up projects inherit **the same tokenizer and checkpoint format**.

- **Kkoma-Chat**: SFT → DPO, chat template, response-only loss, dialogue evaluation
- **Kkoma-Agent**: tool use, planning, multi-step execution, agent runtime (training method
  decided at design time)

The reserved special tokens (`<|reserved_*|>`) receive their meaning when the actual protocol is
fixed in Kkoma-Agent.

---

> **Kkoma-LLM is a compact and reproducible language-model project built to understand the path from
> tokenizer design and architecture exploration to pretraining and Korean language adaptation, while
> providing the foundation for future chat and agent models.**
