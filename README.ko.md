<p align="center">
  <img src="kkoma-mascot-pixel.svg" width="260" alt="둥지에 앉은 꼬마"/>
</p>

<h1 align="center">Kkoma-LLM</h1>

<p align="center"><b><a href="README.md">English</a></b> | 한국어</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.0+"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

> **Kkoma-LLM은 '쓰기 위한' 모델이 아니라 '이해하기 위해' 처음부터 직접 만든 소형 언어 모델입니다.**
> *꼬마* 는 작고 아직 다 자라지 않았지만 직접 만들고 키워 나가는 존재를 뜻합니다. 이 프로젝트가
> 바로 그렇습니다. 거대한 상용 모델을 단순히 축소하거나 남의 가중치를 fine-tuning하는 대신, 제한된 자원
> 안에서 **토크나이저에서 아키텍처, 데이터, 학습으로 이어지는 LLM 개발의 전체 흐름을 끝까지 완주**합니다.
> 목표는 SOTA가 아닙니다. 모든 구성요소를 직접 설계하고, 그 이유를 설명하며, 재현하는 것입니다.

---

## 목차

1. [Kkoma 프로젝트 전체 그림](#1-kkoma-프로젝트-전체-그림)
2. [핵심 설계 철학](#2-핵심-설계-철학)
3. [설치](#3-설치)
4. [저장소 구조](#4-저장소-구조)
5. [빠른 시작 (Quickstart)](#5-빠른-시작-quickstart)
6. [페이즈별 상세 가이드](#6-페이즈별-상세-가이드)
   - [Phase 0 · 토크나이저](#phase-0--토크나이저-preparation)
   - [Phase 1 · 아키텍처 스터디](#phase-1--아키텍처-스터디-architecture-study)
   - [Phase 2 · Base 사전학습](#phase-2--base-사전학습-pretraining)
   - [Phase 3 · 한국어 Continued Pretraining](#phase-3--한국어-continued-pretraining)
7. [평가와 생성](#7-평가와-생성)
8. [Config 다루기](#8-config-다루기)
9. [데이터 디렉터리 규칙](#9-데이터-디렉터리-규칙)
10. [분산 학습 · 체크포인트 · 재현성](#10-분산-학습--체크포인트--재현성)
11. [테스트](#11-테스트)
12. [자주 묻는 질문 (FAQ)](#12-자주-묻는-질문-faq)
13. [v1 범위 밖 / 다음 단계](#13-v1-범위-밖--다음-단계)

---

## 1. Kkoma 프로젝트 전체 그림

Kkoma는 하나의 모델이 아니라 단계적으로 능력을 확장하는 프로젝트 계열입니다.

```
Kkoma-LLM   →   Kkoma-Chat   →   Kkoma-Agent
(이 저장소)      (SFT + DPO)      (tool use, agent)
```

**이 저장소(Kkoma-LLM)** 는 그중 토크나이저와 사전학습 계열만 담당하며, 네 페이즈로 구성됩니다.

| 페이즈 | 단계 | 한 줄 설명 | 대표 산출물 |
|---|---|---|---|
| Phase 0 | Tokenizer Preparation | byte-level BPE, vocab 32,768 | `artifacts/tokenizer/` |
| Phase 1 | Architecture Study | baseline vs modern, 한 번에 한 요소만 변경 | ablation 결과 |
| Phase 2 | Base Pretraining | 125M / 350M / 800M / 1B / 1.3B 처음부터 학습 | `Kkoma-*-Base` |
| Phase 3 | Korean Continued Pretraining | 영어 Base를 한국어에 적응 | `Kkoma-*-Ko-Base` |

SFT/DPO(Kkoma-Chat), tool use·agent(Kkoma-Agent), RL 기반 post-training은 **이 저장소 범위 밖**입니다.

---

## 2. 핵심 설계 철학

- **직접 구현 vs 재사용을 구분한다.** 학습 가치가 높은 모델 구조·학습 루프는 직접 짭니다
  (`kkoma/model`, `kkoma/training`). BPE 알고리즘·CUDA 커널·분산통신 같은 저수준 기능은
  검증된 라이브러리(`tokenizers`, `torch`)를 씁니다. → **`transformers`는 일부러 쓰지 않습니다.**
- **단순하지만 현대적.** nanoGPT 수준의 읽기 쉬운 코드를 유지하면서 RoPE·RMSNorm·SwiGLU·GQA·
  weight tying 같은 현대 구성요소를 config로 갈아끼웁니다.
- **통제된 비교.** 아키텍처 실험은 한 번에 하나의 요소만 바꾸고, 토크나이저·코퍼스·토큰 버짓·
  옵티마이저·시드를 모두 고정합니다.
- **재현 가능성.** config·시드·데이터 manifest(샤드 체크섬 포함)·파라미터 리포트를 체크포인트와
  함께 저장합니다.

---

## 3. 설치

```bash
# 1) 가상환경 (예: venv)
python -m venv .venv && source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt
# 또는 패키지로 설치(권장): kkoma를 어디서든 import 가능
pip install -e .
```

기본 타깃 하드웨어는 **NVIDIA V100 8장 (FP16, NCCL)** 입니다. 단일 GPU·CPU에서도 동작합니다
(CPU 스모크 테스트는 config에서 `precision: fp32` 사용).

> **GPU에서 학습하려면 CUDA 빌드 torch가 필요합니다.** 기본 설치가 CPU 빌드라면 교체하세요.
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

필요 패키지: `torch`, `tokenizers`(BPE 백엔드), `datasets`(FineWeb 스트리밍), `numpy`,
`pyyaml`, `tqdm`. 로깅용 `wandb`와 테스트용 `pytest`는 선택입니다.

---

## 4. 저장소 구조

```text
kkoma-llm/
├── kkoma/                     # 라이브러리 (직접 구현)
│   ├── config.py              #   중첩 dataclass RunConfig (YAML/JSON 로드·저장)
│   ├── tokenizer/             #   BPE 학습·평가·런타임 래퍼·특수토큰
│   ├── model/                 #   RoPE/RMSNorm/SwiGLU/GQA, KV cache, 모델 본체
│   ├── data/                  #   정제·스트리밍·가중 혼합·토큰 패킹
│   ├── training/              #   옵티마이저·스케줄러·DDP·체크포인트·학습 루프
│   ├── generation/            #   샘플링 + KV cache 자기회귀 생성
│   └── evaluation/            #   LM loss(EN/KO)·효율·downstream·생성 샘플
├── configs/                   # 실행 설정 (YAML)
│   ├── tokenizer/             #   tokenizer_32k.yaml
│   ├── architecture/          #   core_125m_* / core_350m_* (누적 progressive, 각 5개)
│   ├── pretraining/           #   base_{125m,350m,800m,1b,1.3b}.yaml
│   └── continued_pretraining/ #   ko_{125m,350m,800m,1b,1.3b}.yaml
├── scripts/                   # CLI: prepare_* / train_* / evaluate / sample
├── tests/                     # 유닛·통합 테스트 (pytest)
└── artifacts/                 # 토크나이저·체크포인트·로그·평가 결과
```

> `scripts/`는 라이브러리가 아니라 실행 폴더라 설치 패키지에 포함되지 않습니다.
> **항상 저장소 루트에서** 실행하세요. 각 스크립트는 루트를 경로에 추가하는 부트스트랩을 갖고 있어
> `python scripts/x.py`와 `python -m scripts.x` 두 방식 모두 동작합니다.

---

## 5. 빠른 시작 (Quickstart)

```bash
# Phase 0: 토크나이저 (혼합 비율·vocab 등은 configs/tokenizer/tokenizer_32k.yaml)
python scripts/prepare_tokenizer_data.py --config configs/tokenizer/tokenizer_32k.yaml \
    --output-dir data/tokenizer
python scripts/train_tokenizer.py --input "data/tokenizer/*.jsonl" \
    --config configs/tokenizer/tokenizer_32k.yaml

# Phase 1: 아키텍처 스터디 (한 번에 한 요소만 변경)
python scripts/prepare_architecture_data.py --output-dir data/architecture --tokenizer artifacts/tokenizer
python scripts/train_architecture.py --config configs/architecture/core_125m_modern.yaml

# Phase 2: Base 사전학습 (단일 GPU)
python scripts/prepare_pretraining_data.py --output-dir data/pretrain --tokenizer artifacts/tokenizer
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml

# Phase 2: 멀티 GPU (한 노드 8장, torchrun)
torchrun --nproc_per_node=8 scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml

# Phase 2: 체크포인트에서 재개 (단일/멀티 GPU 공통, --resume만 추가)
torchrun --nproc_per_node=8 scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml \
    --resume artifacts/checkpoints/base-1b/step_00010000.pt

# Phase 3: 한국어 Continued Pretraining
python scripts/prepare_korean_data.py --output-dir data/korean --tokenizer artifacts/tokenizer
python scripts/train_continued.py --config configs/continued_pretraining/ko_1b.yaml \
    --init-from artifacts/checkpoints/base-1b/final.pt

# 평가 / 생성
python scripts/evaluate.py --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml --output artifacts/evaluation/base_1b.json
python scripts/sample.py --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml --prompt "In the future, small language models"
```

어떤 학습이든 `--resume <checkpoint.pt>`로 이어서 돌릴 수 있습니다(모델·옵티마이저·스케줄러·
grad scaler·스텝·RNG 상태까지 복원).

> **처음 돌릴 때 팁:** 실제 FineWeb 데이터는 매우 큽니다. 파이프라인이 끝까지 도는지 먼저 확인하려면
> `--gb 1`(토크나이저)이나 `--train-tokens 5_000_000`(아키텍처)처럼 작은 값으로 한 번 돌려보세요.

---

## 6. 페이즈별 상세 가이드

### Phase 0 · 토크나이저 (Preparation)

**목적**: 영어·한국어를 모두 합리적인 효율로 처리하고, 이후 Chat·Agent까지 **교체 없이 공유**할
전용 토크나이저를 만든다.

**사양** (spec §4): byte-level BPE, 최종 vocab **정확히 32,768**, UNK 미사용(모든 바이트 표현 가능),
특수토큰 23개(의미 확정 7 + 예약 16).

- **문서 경계·패딩**: `<|bos|>` `<|eos|>` `<|pad|>`
- **Chat 대비 role 토큰** (여기선 vocab에만 포함, 의미 학습 X): `<|system|>` `<|user|>` `<|assistant|>` `<|turn_end|>`
- **미래(Agent) 확장용 예약**: `<|reserved_0|>` … `<|reserved_15|>`

**1) 학습 데이터 샘플링**: 영어 FineWeb-Edu 75% + 한국어 FineWeb2 25%, 약 20GB.

```bash
python scripts/prepare_tokenizer_data.py --config configs/tokenizer/tokenizer_32k.yaml \
    --output-dir data/tokenizer
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config` | (없음) | 혼합 비율·목표 GB를 담은 토크나이저 YAML |
| `--output-dir` | `data/tokenizer` | JSONL 샤드 출력 폴더 |
| `--gb` | config의 `target_gb` 합(없으면 20.0) | 정제 후 목표 크기(바이트 기준) |
| `--seed` | `42` | 샘플링 시드 |
| `--no-progress` | (꺼짐) | 진행바 비활성화 |

> 진행도는 **전체 바(목표 GB·ETA·누적 샤드 수)** + **현재 샤드 바(문서 수)** 두 단계로 표시됩니다.

**2) 토크나이저 학습**

```bash
python scripts/train_tokenizer.py --input "data/tokenizer/*.jsonl" \
    --config configs/tokenizer/tokenizer_32k.yaml
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input` | (필수) | 학습 텍스트 glob (`*.jsonl` 또는 `*.txt`) |
| `--config` | (없음) | vocab·min_frequency·출력 폴더를 담은 토크나이저 YAML |
| `--output-dir` | `artifacts/tokenizer` | 토크나이저 산출물 폴더 (config 덮어씀) |
| `--vocab-size` | `32768` | 특수토큰 포함 최종 크기 (config 덮어씀) |
| `--min-frequency` | `2` | merge 최소 빈도 |

**산출물**: `artifacts/tokenizer/`에 `tokenizer.json`, `vocab.json`, `merges.txt`,
`special_tokens_map.json`, `training_config.json`, `evaluation.json` 등.

**Python에서 직접 써보기**

```python
from kkoma.tokenizer.utils import KkomaTokenizer
tok = KkomaTokenizer.from_file("artifacts/tokenizer")
print(len(tok))                              # 32768
ids = tok.encode("작은 언어 모델을 직접 학습하면", add_bos=True)
print(ids, tok.decode(ids))
```

**완료 조건**: vocab 32,768, 모든 특수토큰이 단일 토큰으로 인코딩, EN/KO round-trip 정상,
GPT-2 대비 한국어 토큰 효율 개선, manifest 저장.

---

### Phase 1 · 아키텍처 스터디 (Architecture Study)

**목적**: 현대 decoder-only 구성요소를 직접 구현하고, **동일 조건**에서 구성요소를 하나씩 누적해
Base에 쓸 최종 구조를 결정한다(spec §6). 실험 규모는 125M과 350M 두 스케일.

**비교 대상**: baseline(learned/LayerNorm/GELU/MHA)에서 시작해 RoPE → RMSNorm → SwiGLU → GQA를
하나씩 **누적**해 modern까지 쌓는 progressive 세트를 두 스케일(`core_125m_*`, `core_350m_*`)로
제공합니다. 각 단계는 직전 단계에서 한 요소만 바뀌므로 그 요소의 효과가 그대로 드러나고, 같은
비교를 두 크기에서 반복해 결론이 스케일에 강건한지 확인합니다. 굵게 표시된 칸이 baseline 대비
켜진(modern) 요소입니다. 모든 config는 토크나이저·코퍼스·토큰 버짓·옵티마이저·시드를 공유하고
`model` 섹션만 다릅니다(spec §5.3 통제된 비교).

| config (`core_125m_*` / `core_350m_*`) | 위치 인코딩 | 정규화 | FFN | 어텐션 |
|---|---|---|---|---|
| `…_baseline` | learned | LayerNorm | GELU | MHA |
| `…_rope` | **RoPE** | LayerNorm | GELU | MHA |
| `…_rope_rmsnorm` | **RoPE** | **RMSNorm** | GELU | MHA |
| `…_rope_rmsnorm_swiglu` | **RoPE** | **RMSNorm** | **SwiGLU** | MHA |
| `…_modern` | **RoPE** | **RMSNorm** | **SwiGLU** | **GQA** |

| 스케일 | n_layer | d_model | n_head | modern n_kv_head | micro_batch | 총 파라미터(≈) |
|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 | 3 | 8 | 99.5M ~ 111M |
| 350M | 24 | 1024 | 16 | 4 | 4 | 304M ~ 342M |

> baseline(MHA·GELU)이 modern(GQA·SwiGLU)보다 파라미터가 큰 것은 정상이다(MHA의 KV projection과
> GELU MLP의 큰 intermediate 때문). modern은 각 Base 모델과 동일한 구조·크기다.

**1) 고정 코퍼스 준비**: FineWeb-Edu 95% + FineWeb2 한국어 5%, 기본 train 300M / val 10M 토큰.

```bash
python scripts/prepare_architecture_data.py \
    --output-dir data/architecture --tokenizer artifacts/tokenizer \
    --train-tokens 300000000 --val-tokens 10000000
```

언어별 디렉터리로 저장되어 학습 시 95:5로 다시 섞이고, 검증 loss는 EN/KO로 따로 계산됩니다
(→ [§9 데이터 디렉터리 규칙](#9-데이터-디렉터리-규칙)).

**2) 변형별 학습** (config만 바꿔 실행)

```bash
# 단일 실행 예: 125M baseline
python scripts/train_architecture.py --config configs/architecture/core_125m_baseline.yaml

# 스케일별 progressive 세트 순차 실행 (각 5개)
for cfg in configs/architecture/core_125m_*.yaml; do
    python scripts/train_architecture.py --config "$cfg"
done
for cfg in configs/architecture/core_350m_*.yaml; do
    python scripts/train_architecture.py --config "$cfg"
done
```

**비교 지표**: train/val loss·perplexity·gradient norm·loss spike(품질), tokens/s·peak memory·
step time·파라미터 수·추정 FLOPs(효율), GQA는 KV cache 크기·생성 지연·생성 tokens/s 추가.

> **SwiGLU 파라미터 공정성:** SwiGLU는 입력 projection이 둘이라, GELU와 블록 파라미터를 맞추려고
> 내부 차원을 `~8/3 × d_model`로 잡고 128 배수로 반올림합니다. (`kkoma/config.py:default_ffn_dim`)

---

### Phase 2 · Base 사전학습 (Pretraining)

**목적**: 확정된 구조로 여러 크기의 Base 모델을 처음부터 학습하고 scaling을 분석한다(spec §13–18).

> **Base 코퍼스는 영어 전용이 아닙니다.** **영어 FineWeb-Edu 95% + 한국어 FineWeb2 5%** 혼합입니다
> (spec §14.1). "영어 중심 Base"는 영어가 압도적이라는 뜻이지 한국어가 0%라는 뜻이 아닙니다. 5%는
> 한글 토큰 임베딩을 미리 데워 두어 Phase 3가 백지에서 출발하지 않게 합니다. 따라서
> `prepare_pretraining_data.py`는 영어·한국어를 **둘 다** 내려받습니다(한국어는 5% 분량만).

| 모델 | n_layer | d_model | n_head | n_kv_head | 토큰 버짓 | peak LR | config |
|---|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12 | 3 | 2.5B | 6e-4 | `base_125m.yaml` |
| 350M | 24 | 1024 | 16 | 4 | 7B | 3e-4 | `base_350m.yaml` |
| 800M | 24 | 1536 | 24 | 6 | 16B | 2e-4 | `base_800m.yaml` |
| 1B | 24 | 1792 | 28 | 7 | 18B | 1.8e-4 | `base_1b.yaml` |
| 1.3B | 24 | 2048 | 32 | 8 | 23B | 1.5e-4 | `base_1.3b.yaml` |

> 모델 이름은 근사치입니다. vocab 32K + weight tying을 포함한 **실제 파라미터 수**는 학습 시작 시
> `parameter_report`로 저장됩니다(예: base_125m → 약 99.5M, base_800m → 약 645M, base_1b → 약 879M,
> base_1.3b → 약 1130M).
>
> **모두 V100 32GB에 들어갑니다**(DDP는 GPU마다 모델을 복제). rank당 실측 peak(seq 1024, fp16 + AdamW):
> base_1b `micro_batch_size: 2` → 18.4 GB, base_1.3b `micro_batch_size: 2` → 22.6 GB(둘 다 config
> 기본값). base_1.3b는 상한에 가깝습니다. activation checkpointing 없는 실질 한계는 약 1.5B입니다.

**1) 코퍼스 준비**

```bash
python scripts/prepare_pretraining_data.py \
    --output-dir data/pretrain --tokenizer artifacts/tokenizer --tokens 23e9
```

이 명령은 영어(95%)와 한국어(5%)를 **둘 다** 스트리밍해 언어별 디렉터리로 씁니다:
`train/` + `val/`(영어), `train_ko/` + `val_ko/`(한국어). 각 언어에서 필요한 분량만 받으므로
`--tokens 23e9` 기준 한국어는 FineWeb2 전체가 아니라 약 1.15B 토큰(5%)만 내려받습니다.
23B 코퍼스를 한 번 만들면 125M은 앞 2.5B, 350M은 앞 7B, 800M은 앞 16B, 1B는 앞 18B, 1.3B는
전체를 사용합니다(실제로 읽는 양은 각 config의 `training.max_tokens`가 결정).

**2) 학습 (단일 GPU)**

```bash
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml
```

**2′) 학습 (멀티 GPU, V100 8장)**

```bash
torchrun --nproc_per_node=8 scripts/train_pretraining.py \
    --config configs/pretraining/base_1b.yaml
```

**배치 토큰 계산** (spec §15.4): global batch는 약 256K 토큰을 목표로 합니다.

```
global_batch_tokens = micro_batch_size × sequence_length × grad_accum_steps × world_size
예) 4 × 1024 × 8 × 8 = 262,144
```

`grad_accum_steps`를 비워두면 `global_batch_tokens`에서 자동 계산됩니다.

**중단 후 재개**

```bash
python scripts/train_pretraining.py --config configs/pretraining/base_1b.yaml \
    --resume artifacts/checkpoints/base-1b/step_00010000.pt
```

체크포인트는 `step_00010000.pt`(주기), `best_val.pt`(최저 검증 loss), `final.pt`(종료)로 저장됩니다.
각 체크포인트에는 재현성 식별자(git commit, 토크나이저 sha256, data manifest sha256, W&B run id)가
함께 기록됩니다(spec §18.1, §25).

재개 시 W&B run id를 체크포인트에서 읽어 **같은 run을 이어서** 기록하고(spec §23), 데이터
스트림은 소비한 블록 수만큼 **fast-forward**되어 중단 없이 이어진 것과 동일한 다음 배치부터
계속됩니다(spec §18.3). fast-forward는 건너뛰는 구간을 한 번 재토크나이즈하므로 큰 코퍼스에서는
시간이 걸릴 수 있고, `training.resume_fastforward: false`로 끄면 스트림을 처음부터 다시 읽되
그 사실이 체크포인트에 기록됩니다.

**W&B 프로젝트**는 학습 단계별로 분리됩니다. 같은 단계의 모든 크기는 하나의 프로젝트를 공유하고,
크기·구성 구분은 run name(`base-1b-tpp20`, `ko-1b-cpt-2b` 등)으로 합니다. `project.name`은 항상
`kkoma-llm`으로 두고, `logging.project`만 다릅니다.

| 단계 | Config | W&B 프로젝트 |
|---|---|---|
| Base pretraining | `configs/pretraining/base_*.yaml` | `kkoma-llm-pt` |
| Continued pretraining | `configs/continued_pretraining/ko_*.yaml` | `kkoma-llm-cpt` |
| Architecture 실험 | `configs/architecture/core_*.yaml` | `kkoma-llm-core` |

---

### Phase 3 · 한국어 Continued Pretraining

**목적**: 영어 중심 Base(사전학습 시 영어 95% / 한국어 5%)를 한국어 위주 혼합으로 적응시키되
영어 능력 유지·forgetting을 측정한다(spec §19). 입력: `Kkoma-<크기>-Base` → 출력: `Kkoma-<크기>-Ko-Base`.

**기본 설정**: 한국어 70% + 영어 replay 30%, 2B 토큰(≈KO 1.4B / EN 0.6B), peak LR 5e-5.
같은 레시피가 모든 Base 크기에 적용됩니다. `ko_125m` / `ko_350m` / `ko_800m` / `ko_1b` /
`ko_1.3b`는 모델 dims만 다르고 각 `base_*`와 동일합니다. 아래 예시는 1B 기준입니다.
**v1 기본안은 새 옵티마이저 상태로 시작**합니다(즉 `--resume` 없이 `--init-from`만 사용).

**1) 코퍼스 준비**

```bash
python scripts/prepare_korean_data.py \
    --output-dir data/korean --tokenizer artifacts/tokenizer --tokens 2e9
```

**2) Base 가중치로 시작해 한국어 학습**

```bash
python scripts/train_continued.py \
    --config configs/continued_pretraining/ko_1b.yaml \
    --init-from artifacts/checkpoints/base-1b/final.pt
```

| 옵션 | 설명 |
|---|---|
| `--init-from` | (필수) 적응시킬 Base 체크포인트. **가중치만** 로드(옵티마이저는 새로 시작) |
| `--resume` | 중단된 CPT 자체를 이어서 재개(전체 상태 복원) |

**forgetting 측정**: 검증 loaders가 `en`/`ko`로 분리돼 있어 학습 중 영어/한국어 loss가 따로 찍힙니다.
`val/loss_en`이 학습 전 Base 대비 얼마나 올라가는지로 catastrophic forgetting을 봅니다.

---

## 7. 평가와 생성

**체크포인트 평가**: LM loss(EN/KO 분리), 효율 벤치마크, 고정 프롬프트 생성 샘플을 JSON으로 저장.

```bash
python scripts/evaluate.py \
    --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml \
    --output artifacts/evaluation/base_1b.json
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--max-batches` | `50` | 검증에 사용할 배치 수 |
| `--no-generation` | (꺼짐) | 생성 샘플 단계 건너뛰기 |

**텍스트 생성**: KV cache 자기회귀 생성. 같은 시드면 결과가 재현됩니다.

```bash
python scripts/sample.py \
    --checkpoint artifacts/checkpoints/base-1b/final.pt \
    --config configs/pretraining/base_1b.yaml \
    --prompt "대한민국의 수도는" \
    --max-new-tokens 100 --temperature 0.8 --top-k 50
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--max-new-tokens` | `100` | 생성 토큰 수 |
| `--temperature` | `0.8` | 0이면 greedy |
| `--top-k` / `--top-p` | `50` / `1.0` | 샘플링 필터 |
| `--seed` | `1234` | 재현용 시드 |

**고정 평가 프롬프트** (모든 체크포인트에서 동일 시드로 비교):
영어 `"The meaning of life is"`, `"Artificial intelligence can"`,
`"In the future, small language models"` / 한국어 `"인공지능이란"`, `"대한민국의 수도는"`,
`"작은 언어 모델을 직접 학습하면"`.

---

## 8. Config 다루기

모든 실행은 YAML 하나로 정의됩니다(`RunConfig`). 핵심은 `model` 섹션입니다. 여기 한 줄만 바꾸면
다른 아키텍처가 됩니다.

```yaml
model:
  vocab_size: 32768
  context_length: 1024
  n_layer: 12
  d_model: 768
  n_head: 12
  n_kv_head: 3              # n_head와 같으면 MHA, 작으면 GQA
  norm: rmsnorm            # rmsnorm | layernorm
  positional_encoding: rope # rope | learned
  activation: swiglu       # swiglu | gelu
  tie_word_embeddings: true
  bias: false
  dropout: 0.0

training:
  precision: fp16          # CPU 테스트는 fp32
  global_batch_tokens: 262144
  micro_batch_size: 4
  max_tokens: 2500000000

optimizer: { name: adamw, learning_rate: 0.0006, beta1: 0.9, beta2: 0.95, weight_decay: 0.1 }
scheduler: { name: cosine, warmup_ratio: 0.02, min_lr_ratio: 0.1 }
```

코드에서 직접 만들거나 검사하기:

```python
from kkoma.config import RunConfig
cfg = RunConfig.from_yaml("configs/pretraining/base_1b.yaml")

from kkoma.model.model import KkomaModel
m = KkomaModel(cfg.model)
print(m.parameter_report()["total"])        # 실제 파라미터 수
```

---

## 9. 데이터 디렉터리 규칙

`prepare_*` 스크립트는 **언어별 디렉터리**로 데이터를 씁니다. 이렇게 하면 (1) 학습 시 config의
가중치로 혼합 비율을 통제하고, (2) **EN/KO 검증 loss를 따로** 계산할 수 있습니다(spec §14.4).

```text
data/pretrain/
├── train/        # 영어 (train 버짓의 95%)   → config에서 weight 0.95
├── train_ko/     # 한국어 (train 버짓의 5%)   → config에서 weight 0.05
├── val/          # 영어 검증                  → val/loss_en
└── val_ko/       # 한국어 검증                → val/loss_ko
```

학습 코퍼스는 prepare 단계에서 비율대로 나눠 두고, **로드 시 `MixtureStream`이 시드 기반으로 섞습니다.**
한국어 CPT는 `train_ko`(70%) / `train_en`(30%) / `val_ko` / `val_en` 구조를 씁니다.

**train/val 분리는 문서 단위 해시 holdout**입니다(spec §14.3). 각 문서는
`sha256(text) % holdout_mod == 0`이면 검증 풀로 배정되므로(기본 1%), 스트림 순서나 시드와
무관하게 train과 val이 겹칠 수 없습니다. 문서 순서는 prepare 단계에서 시드 기반으로 셔플되어
고정됩니다(spec §14.2, `--shuffle-buffer`, 기본 10,000).

`data_manifest.json`에는 소스·시드·셔플 설정·holdout 방식·문서 수·토큰 수·샤드 SHA-256
체크섬이 기록됩니다.

---

## 10. 분산 학습 · 체크포인트 · 재현성

- **DDP**: GPU당 한 프로세스, NCCL 백엔드. `torchrun --nproc_per_node=N`으로 실행하면 코드가
  환경변수(`RANK`/`WORLD_SIZE`/`LOCAL_RANK`)를 읽어 자동 구성합니다. gradient accumulation 중에는
  `no_sync()`로 불필요한 all-reduce를 피하고 마지막 micro-step에서만 동기화합니다.
- **혼합정밀(FP16)**: autocast + GradScaler. 매 로그 주기에 loss finite 여부·grad norm·scaler scale·
  skipped step을 점검하고, NaN/Inf면 스텝을 건너뜁니다.
- **체크포인트**: 모델·옵티마이저·스케줄러·scaler·global step·처리 토큰·RNG·config·식별자를 저장.
  weight tying은 로드 후에도 동일 Parameter 객체로 복원됩니다. 최신 N개만 유지 + best 별도 보관.
- **재현성**: global/data/init/sampling 시드를 분리 관리하고 config와 함께 저장. 데이터 manifest·
  파라미터 리포트·시드가 남아 동일 조건 재현이 가능합니다.

---

## 11. 테스트

```bash
pytest                 # 전체
pytest tests/test_kv_cache.py -v   # 일부만
```

커버리지: 토크나이저 round-trip / 특수토큰 단일 ID / vocab 크기, RoPE shape·offset·결정성,
GQA head 매핑, causal masking, RMSNorm 수치, SwiGLU shape, **weight tying(저장·로드 후 유지)**,
**KV cache 동치(cache on/off greedy 결과 일치)**, 체크포인트 resume, 결정성, 소형 overfit,
스케줄러 warmup/decay 형태.

---

## 12. 자주 묻는 질문 (FAQ)

**`ModuleNotFoundError: No module named 'kkoma'`**
저장소 루트에서 실행하지 않았을 때 납니다. 루트에서 `python scripts/x.py`로 실행하거나
`pip install -e .` 후 사용하세요.

**`transformers`는 안 쓰나요?**
네. 모델·생성·학습 루프를 직접 구현하는 게 프로젝트 목적이라 일부러 제외했습니다.
HF 라이브러리는 `tokenizers`(BPE 백엔드)와 `datasets`(스트리밍)만 씁니다.

**데이터는 전부 미리 다운로드되나요?**
아니요. `datasets`를 `streaming=True`로 써서 소비하는 만큼만 네트워크로 가져옵니다. 진행바는
**로컬 샤드로 기록되는 양**을 기준으로 하므로 실제 진척도와 일치합니다.

**CPU에서 돌려봐도 되나요?**
파이프라인 점검용으로는 됩니다. config에서 `precision: fp32`로 두고 작은 토큰 버짓을 쓰세요.
실제 학습은 V100 등 GPU + CUDA 빌드 torch가 필요합니다.

**꼭 토크나이저부터 만들어야 하나요?**
네. 학습/평가 config가 `tokenizer.path: artifacts/tokenizer`를 참조하므로 Phase 0이 선행돼야 합니다.

**`kkoma_llm.egg-info/`는 뭔가요?**
`pip install -e .`가 만든 설치 메타데이터입니다. `.gitignore` 처리돼 있고, 지워도 다시 생깁니다.

---

## 13. v1 범위 밖 / 다음 단계

이 저장소(Kkoma-LLM v1)에서 **제외**: SFT·DPO, PPO/GRPO/RLHF, tool use·function calling·agent,
long-context 사전학습, Mixture-of-Experts, 대규모 synthetic data.

다음 프로젝트는 **동일 토크나이저와 체크포인트 형식**을 이어받습니다.

- **Kkoma-Chat**: SFT → DPO, chat template, response-only loss, 대화 평가
- **Kkoma-Agent**: tool use, planning, multi-step 실행, agent runtime (학습 방식은 설계 단계에서 결정)

예약 특수토큰(`<|reserved_*|>`)은 Kkoma-Agent에서 실제 프로토콜이 확정될 때 의미가 부여됩니다.

---

> **Kkoma-LLM은 토크나이저 설계와 아키텍처 탐구에서 사전학습, 한국어 적응으로 이어지는 과정을
> 이해하기 위해 만든, 작고 재현 가능한 언어 모델 프로젝트이며, 이후의 chat·agent 모델을 위한
> 기반을 제공합니다.**
