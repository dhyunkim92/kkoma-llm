# Kkoma-LLM Implementation Specification

## 1. 문서 목적

이 문서는 Kkoma-LLM v1의 구현 기준을 정의한다.

`design.md`가 프로젝트의 목적과 범위를 설명한다면, 본 문서는 다음 사항을 구체적으로 정의한다.

- 토크나이저 사양
- 모델 아키텍처
- 모델 크기별 구성
- 데이터셋 및 데이터 혼합
- 학습 레시피
- 분산 학습과 체크포인트
- 추론 및 KV cache
- 평가 절차
- 실험 재현성
- 저장소 구조
- 단계별 완료 조건

수치는 초기 기본값이며 pilot run 결과에 따라 조정할 수 있다. 단, 변경 시 config와 변경 이유를 기록해야 한다.


### 1.1 전체 프로젝트 단계와 현재 범위

```text
Kkoma-LLM
- Preparation
- Pre-training
- Mid-training

Kkoma-Chat
- SFT
- DPO

Kkoma-Agent
- Agent system and runtime
- 상세 학습 레시피는 추후 결정
```

Kkoma-Chat v1은 `SFT → DPO`까지만 수행한다. PPO, GRPO, RLHF 및 기타 RL 기반 post-training은 현재 구현 범위에 포함하지 않는다.

---

## 2. 지원 환경

### 2.1 기본 하드웨어

```text
GPU: NVIDIA V100 8장
Distributed backend: NCCL
Primary precision: FP16
Primary framework: PyTorch
```

### 2.2 권장 소프트웨어

```text
Python: 3.11
PyTorch: 2.x
CUDA: 설치된 V100 환경과 호환되는 버전
Tokenizer: Hugging Face tokenizers
Dataset streaming: Hugging Face datasets 또는 자체 JSONL/Parquet loader
Logging: Weights & Biases
Configuration: Python config 또는 YAML
```

V100에서는 BF16을 기본으로 사용하지 않는다. 공식 FlashAttention-2 사용도 필수 요구사항으로 두지 않는다.

---

## 3. 저장소 구조

```text
kkoma-llm/
├── README.md
├── README.ko.md
├── kkoma-mascot.svg
├── pyproject.toml
├── requirements.txt
│
├── docs/
│   ├── design.md
│   └── spec.md
│
├── kkoma/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── tokenizer/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── special_tokens.py
│   │   └── utils.py
│   │
│   ├── model/
│   │   ├── model.py
│   │   ├── attention.py
│   │   ├── positional.py
│   │   ├── normalization.py
│   │   ├── feedforward.py
│   │   ├── cache.py
│   │   └── initialization.py
│   │
│   ├── data/
│   │   ├── streaming.py
│   │   ├── preprocessing.py
│   │   ├── packing.py
│   │   └── mixture.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── distributed.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpoint.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── language_modeling.py
│   │   ├── generation.py
│   │   ├── efficiency.py
│   │   └── downstream.py
│   │
│   └── generation/
│       ├── generate.py
│       └── sampling.py
│
├── configs/
│   ├── tokenizer/
│   ├── architecture/
│   ├── pretraining/
│   └── continued_pretraining/
│
├── scripts/
│   ├── prepare_tokenizer_data.py
│   ├── train_tokenizer.py
│   ├── prepare_architecture_data.py
│   ├── train_architecture.py
│   ├── prepare_pretraining_data.py
│   ├── train_pretraining.py
│   ├── prepare_korean_data.py
│   ├── train_continued.py
│   ├── evaluate.py
│   └── sample.py
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_rope.py
│   ├── test_gqa.py
│   ├── test_weight_tying.py
│   ├── test_kv_cache.py
│   ├── test_checkpoint_resume.py
│   └── test_determinism.py
│
└── artifacts/
    ├── tokenizer/
    ├── checkpoints/
    ├── logs/
    └── evaluation/
```

---

## 4. Phase 0: 토크나이저 구현 스펙

### 4.1 기본 방식

```text
Algorithm: Byte-level BPE
Implementation: Hugging Face tokenizers의 Rust backend
Final vocabulary size: 32,768
Normalization: 최소화
Unknown token: 사용하지 않거나 비활성화
Byte coverage: 모든 입력을 표현 가능해야 함
```

토크나이저 알고리즘 자체는 새로 구현하지 않는다. Kkoma 데이터로 vocabulary와 merge rules를 직접 학습한다.

### 4.2 학습 데이터

```text
English: FineWeb-Edu 75%
Korean: FineWeb2 Korean 25%
Recommended sample size: 총 10~30GB
Initial target: 약 20GB
```

권장 초기 구성:

```text
English: 15GB
Korean: 5GB
```

비율은 문서 수가 아니라 정제 후 실제 byte 또는 character 양을 기준으로 맞춘다.

### 4.3 전처리

다음 항목만 최소한으로 수행한다.

- 빈 문서 제거
- 지나치게 짧은 문서 제거
- NUL 및 비정상 제어 문자 제거
- UTF-8 decode 실패 문서 제거
- 문서 단위 중복 최소화
- 영어·한국어 비율에 맞춘 seed 기반 샘플링

과도한 Unicode normalization이나 한국어 형태소 분석은 적용하지 않는다.

### 4.4 특수 토큰

공식 특수 토큰:

```text
<|bos|>
<|eos|>
<|pad|>
<|system|>
<|user|>
<|assistant|>
<|turn_end|>
```

예약 토큰:

```text
<|reserved_0|>
<|reserved_1|>
...
<|reserved_15|>
```

총 예약 및 특수 토큰 수:

```text
7개의 의미 확정 토큰
16개의 미래 확장 토큰
총 23개
```

최종 tokenizer 길이는 특수 토큰을 포함해 정확히 32,768이어야 한다.

```python
assert len(tokenizer) == 32768
```

`vocab_size=32768` 설정 시 라이브러리가 특수 토큰을 포함하는지 반드시 확인한다.

### 4.5 특수 토큰 사용 규칙

Base pretraining:

```text
<|bos|> document text <|eos|>
```

Kkoma-Chat의 예정 형식:

```text
<|bos|>
<|system|>
system content
<|turn_end|>
<|user|>
user content
<|turn_end|>
<|assistant|>
assistant content
<|turn_end|>
<|eos|>
```

Kkoma-LLM에서는 role token을 vocabulary에만 포함하고, role 의미 학습은 수행하지 않는다.

### 4.6 토크나이저 평가

비교 대상:

```text
GPT-2 tokenizer
Kkoma-Tokenizer-32K
```

선택 비교:

```text
Kkoma-Tokenizer-48K
```

필수 지표:

- 영어 tokens per word
- 한국어 tokens per eojeol
- tokens per character
- 문서별 평균 sequence length
- 영어/한국어 compression ratio
- byte token 사용 비율
- vocabulary usage frequency
- embedding parameter 수

정성 평가 문장:

```text
The model is trained from scratch.

이 모델은 처음부터 직접 학습되었습니다.

Kkoma-LLM은 영어와 한국어를 모두 처리합니다.

def train_model(config):
    return model
```

### 4.7 저장 파일

```text
artifacts/tokenizer/
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
├── merges.txt
├── training_config.json
├── data_manifest.json
└── evaluation.json
```

---

## 5. 모델 공통 아키텍처

### 5.1 모델 형식

```text
Architecture: Decoder-only Transformer
Objective: Next-token prediction
Attention mask: Causal
Normalization placement: Pre-Norm
Embedding tying: Enabled
Linear bias: Disabled
Dropout: 0.0
```

### 5.2 블록 구조

```text
x
├── x + Attention(Norm(x))
└── x + FFN(Norm(x))
```

각 Transformer block:

1. RMSNorm 또는 LayerNorm
2. Causal self-attention
3. Residual addition
4. RMSNorm 또는 LayerNorm
5. GELU MLP 또는 SwiGLU FFN
6. Residual addition

### 5.3 최종 공식 구조 후보

Architecture Study 결과 전까지의 기본 후보:

```text
RoPE
RMSNorm
SwiGLU
GQA
Pre-Norm
Bias-free Linear
Tied Embeddings
```

GQA가 소형 모델에서 유의미한 성능 저하를 보이면 MHA를 유지할 수 있다.

---

## 6. Architecture Study 스펙

### 6.1 기준 모델

공식 구조 비교는 125M과 350M 두 스케일에서 수행한다. 같은 스케일 안에서는 모든 변형이 동일한 dims(13장 참조)를 공유하고, 실제 총 파라미터는 vocabulary embedding 크기의 영향을 포함해 config 생성 시 계산한다.

Baseline:

```text
Learned Positional Embedding
LayerNorm
GELU
MHA
Pre-Norm
Bias-free
Tied embeddings
```

Modern:

```text
RoPE
RMSNorm
SwiGLU
GQA
Pre-Norm
Bias-free
Tied embeddings
```

### 6.2 실험 모델

Baseline에서 구성 요소를 하나씩 누적해 Modern까지 쌓는 progressive 세트를 두 스케일로 학습한다.

```text
Kkoma-Core-{125M,350M}-Baseline
Kkoma-Core-{125M,350M}-RoPE
Kkoma-Core-{125M,350M}-RoPE-RMSNorm
Kkoma-Core-{125M,350M}-RoPE-RMSNorm-SwiGLU
Kkoma-Core-{125M,350M}-Modern
```

각 단계는 직전 단계에서 한 요소만 변경하므로 그 요소의 효과가 그대로 드러나고, 두 스케일에서 같은 비교를 반복해 결론이 스케일에 강건한지 확인한다.

### 6.3 파라미터 공정성

SwiGLU는 두 개의 input projection을 사용하므로 GELU MLP와 같은 intermediate size를 쓰면 파라미터가 늘어난다.

따라서 비교 시 intermediate size를 조정해 전체 블록 파라미터 수를 최대한 맞춘다.

권장 근사:

```text
GELU FFN intermediate size: 4 × d_model
SwiGLU intermediate size: 약 8/3 × d_model
```

실제 값은 tensor core 효율을 위해 64 또는 128의 배수로 반올림한다.

### 6.4 MHA와 GQA

MHA:

```text
num_key_value_heads = num_attention_heads
```

GQA 기본 후보:

```text
num_key_value_heads = num_attention_heads / 4
```

예:

```text
125M: num_attention_heads = 12, num_key_value_heads = 3
350M: num_attention_heads = 16, num_key_value_heads = 4
```

조건:

- `num_attention_heads % num_key_value_heads == 0`
- query head가 KV head에 균등하게 매핑되어야 함

### 6.5 공식 구조 비교 데이터

```text
FineWeb-Edu English: 95%
FineWeb2 Korean: 5%
Train budget: 200M~500M tokens
Validation: 10M~20M tokens
Context length: 512 또는 1,024
```

권장 초기값:

```text
Train: 300M tokens
Validation: 10M tokens
Context length: 1,024
```

### 6.6 Smoke test

```text
Dataset: TinyStories
Purpose:
- forward/backward 검증
- loss 감소 확인
- generation 확인
- overfit test
```

각 모듈은 소규모 batch 또는 단일 문서에 overfit할 수 있어야 한다.

### 6.7 비교 지표

학습 품질:

- train loss
- validation loss
- perplexity
- gradient norm
- loss spike 횟수

효율:

- tokens per second
- peak GPU memory
- step time
- parameter count
- estimated FLOPs

GQA 추가 지표:

- KV cache 크기
- generation latency
- generation tokens per second

---

## 7. Position Encoding 스펙

### 7.1 Learned Position Embedding

```text
Shape: [max_seq_len, d_model]
Added to token embedding
Used only in baseline comparison
```

### 7.2 RoPE

```text
Applied to Q and K
Default base theta: 10,000
Training context: 1,024
Long-context scaling: 사용하지 않음
```

RoPE cache는 최대 context length와 head dimension에 맞춰 미리 생성하거나 동적으로 확장한다.

테스트:

- shape 보존
- 동일 position에서 deterministic
- offset position 지원
- KV cache generation과 일치

---

## 8. Normalization 스펙

### 8.1 LayerNorm

Baseline 비교에 사용한다.

```text
Affine: True
Bias: 구현 옵션
Default baseline bias: True 또는 PyTorch 기본
```

### 8.2 RMSNorm

최종 구조 후보에 사용한다.

```text
Epsilon: 1e-5 또는 1e-6
Default: 1e-5
Learnable scale: True
Bias: 없음
```

RMSNorm은 FP32 accumulation을 사용한 뒤 원래 dtype으로 되돌리는 구현을 권장한다.

---

## 9. Feed-Forward Network 스펙

### 9.1 GELU MLP

```text
Linear(d_model, 4*d_model)
GELU
Linear(4*d_model, d_model)
```

### 9.2 SwiGLU

```text
gate = Linear(d_model, d_ff)
value = Linear(d_model, d_ff)
hidden = SiLU(gate) * value
output = Linear(d_ff, d_model)
```

`d_ff`는 파라미터 수를 맞추기 위해 약 `8/3 × d_model`을 사용하고 64 또는 128 배수로 반올림한다.

---

## 10. Attention 스펙

### 10.1 기본 구현

```text
API: torch.nn.functional.scaled_dot_product_attention
Mask: is_causal=True
Dropout: 0.0
```

V100에서는 FlashAttention-2 사용을 전제로 하지 않는다.

### 10.2 Projection

Bias-free projection:

```text
q_proj
k_proj
v_proj
o_proj
```

가능하면 fused QKV projection도 구현 옵션으로 제공할 수 있지만, 첫 구현은 분리 projection을 허용한다.

### 10.3 Tensor shape

MHA:

```text
Q: [batch, n_heads, seq, head_dim]
K: [batch, n_heads, seq, head_dim]
V: [batch, n_heads, seq, head_dim]
```

GQA:

```text
Q: [batch, n_heads, seq, head_dim]
K: [batch, n_kv_heads, seq, head_dim]
V: [batch, n_kv_heads, seq, head_dim]
```

attention 계산 시 KV head를 query group에 맞게 broadcast 또는 repeat한다.

---

## 11. Weight Tying 스펙

```python
self.lm_head.weight = self.token_embedding.weight
```

요구사항:

- 동일한 Parameter 객체를 참조해야 함
- checkpoint save/load 후에도 tying이 유지되어야 함
- optimizer parameter list에 중복 등록되지 않아야 함

테스트:

```python
assert model.lm_head.weight is model.token_embedding.weight
```

---

## 12. 초기화 스펙

기본 초기화:

```text
Embedding and Linear weights:
Normal(mean=0.0, std=0.02)

Bias:
사용 시 zero initialization

Residual output projections:
std = 0.02 / sqrt(2 * n_layer)
```

Residual scaling 대상:

- attention output projection
- FFN output projection

다양한 initialization 비교는 Kkoma v1 범위에서 제외한다.

---

## 13. Base 모델 크기 스펙

정확한 구성은 vocabulary 32K와 weight tying을 포함한 실제 파라미터 계산으로 확정한다.

초기 후보:

### 13.1 Kkoma-125M-Base

```text
n_layer: 12
d_model: 768
n_head: 12
n_kv_head: 3 또는 12
head_dim: 64
d_ff: SwiGLU 기준 약 2,048
context_length: 1,024
vocab_size: 32,768
```

### 13.2 Kkoma-350M-Base

```text
n_layer: 24
d_model: 1,024
n_head: 16
n_kv_head: 4 또는 16
head_dim: 64
d_ff: SwiGLU 기준 약 2,752
context_length: 1,024
vocab_size: 32,768
```

### 13.3 Kkoma-800M-Base

초기 후보 A:

```text
n_layer: 24
d_model: 1,536
n_head: 24
n_kv_head: 6 또는 24
head_dim: 64
d_ff: SwiGLU 기준 약 4,096
context_length: 1,024
vocab_size: 32,768
```

실제 파라미터 수가 목표와 크게 다르면 `n_layer`, `d_model`, `d_ff`를 조정한다.

### 13.4 Kkoma-1B-Base

```text
n_layer: 24
d_model: 1,792
n_head: 28
n_kv_head: 7
head_dim: 64
d_ff: SwiGLU 기준 약 4,864
context_length: 1,024
vocab_size: 32,768
```

실제 파라미터는 약 879M(임베딩 포함)이다. V100 32GB 한 장에 여유 있게 들어간다. rank당 실측 peak(seq 1,024, fp16 autocast + AdamW): `micro_batch_size=2` → 18.4 GB(기본값), 1 → 15.4 GB, 4 → 24.0 GB. `micro_batch_size=2`에서 grad_accum이 16으로 유도된다(world_size 8, 262,144 토큰 global batch).

### 13.5 Kkoma-1.3B-Base

```text
n_layer: 24
d_model: 2,048
n_head: 32
n_kv_head: 8
head_dim: 64
d_ff: SwiGLU 기준 약 5,504
context_length: 1,024
vocab_size: 32,768
```

실제 파라미터는 약 1,130M(임베딩 포함)이다. rank당 실측 peak: `micro_batch_size=2` → 22.6 GB(기본값, grad_accum 16), 1 → 19.3 GB. token budget 11.5B(tpp ~10), peak LR 1.5e-4. activation checkpointing 없이 V100 32GB에 올릴 수 있는 실질 상한(약 1.5B)에 가깝다.

모든 이름의 `125M`, `350M`, `800M`, `1B`, `1.3B`은 실제 파라미터 수와 충분히 가까워야 한다. 최종 config 생성 시 parameter count report를 저장한다.

---

## 14. Pretraining 데이터 스펙

### 14.1 기본 mixture

```text
FineWeb-Edu English: 95%
FineWeb2 Korean: 5%
```

Base pretraining은 영어 전용이 아니다. 한국어가 5% 포함되며(0%가 아님), 이 5%는 토크나이저에 이미 존재하는 한글 토큰 임베딩을 최소한으로 학습시켜 Phase 3 continued pretraining이 백지에서 출발하지 않게 한다. 따라서 `prepare_pretraining_data.py`는 영어와 한국어를 모두 내려받는다(한국어는 5% 분량만).

### 14.2 Master corpus

총 11.5B-token master corpus를 seed 기반으로 생성한다. corpus는 영어(95%)와 한국어(5%) 두 언어를 모두 포함하며, 언어별 디렉터리(`train/`·`train_ko/`·`val/`·`val_ko/`)로 저장해 학습 시 config weight로 혼합하고 검증 loss를 언어별로 계산한다.

```text
앞 1.25B tokens → Kkoma-125M
앞 3.5B tokens  → Kkoma-350M
앞 8B tokens    → Kkoma-800M
앞 9B tokens    → Kkoma-1B
전체 11.5B tokens → Kkoma-1.3B
```

corpus 생성 전에 문서 순서를 seed 기반으로 섞고 고정한다.

### 14.3 문서 처리

- 문서별 `<|bos|>`와 `<|eos|>` 삽입
- 문서 간 경계를 보존한 packing
- sequence를 context length로 packing
- pad 사용을 최소화
- 문서 단위 validation holdout
- train/validation URL 또는 document ID 중복 방지

### 14.4 Validation set

```text
English validation: 10M~20M tokens
Korean validation: 5M~10M tokens
```

영어와 한국어 loss를 별도로 계산한다.

---

## 15. Base Pretraining 레시피

### 15.1 공통 설정

```text
Context length: 1,024
Precision: FP16
Optimizer: AdamW
Betas: (0.9, 0.95)
Weight decay: 0.1
Gradient clipping: 1.0
Warmup: 총 step의 2%
Scheduler: Cosine decay
Minimum LR: Peak LR의 10%
Dropout: 0.0
```

### 15.2 Token budget

```text
Kkoma-125M: 1.25B tokens
Kkoma-350M: 3.5B tokens
Kkoma-800M: 8B tokens
Kkoma-1B:   9B tokens
Kkoma-1.3B: 11.5B tokens
```

### 15.3 초기 peak learning rate

```text
125M:  6e-4
350M:  3e-4
800M:  2e-4
1B:    1.8e-4
1.3B:  1.5e-4
```

각 모델은 본 학습 전에 짧은 pilot run을 수행한다.

Pilot 확인 항목:

- 초기 loss 수준
- gradient norm
- NaN/Inf
- loss spike
- 처리량
- 메모리 여유

### 15.4 Global batch

초기 목표:

```text
Global batch size: 약 256K tokens
```

계산:

```text
global_batch_tokens
= batch_size_per_gpu
× sequence_length
× gradient_accumulation_steps
× world_size
```

예:

```text
batch_size_per_gpu = 4
sequence_length = 1024
gradient_accumulation_steps = 8
world_size = 8

global_batch_tokens = 262,144
```

모델 크기에 따라 micro-batch는 줄이되 global batch token 수는 가능한 한 유지한다.

---

## 16. 분산 학습 스펙

### 16.1 기본 방식

```text
Distributed Data Parallel
Backend: NCCL
One process per GPU
```

### 16.2 Gradient accumulation

- accumulation 중에는 불필요한 all-reduce를 피하기 위해 `no_sync()` 사용
- 마지막 micro-step에서만 gradient synchronization
- optimizer step 단위로 global step 증가

### 16.3 Random seed

다음 seed를 분리 관리한다.

```text
global seed
data shuffle seed
model initialization seed
sampling seed
```

각 run의 config와 함께 저장한다.

---

## 17. Mixed Precision 및 안정성

### 17.1 FP16

```text
Autocast: enabled
Gradient scaler: enabled
```

### 17.2 안정성 검사

매 log interval마다 다음을 확인한다.

- loss finite 여부
- gradient norm
- scaler scale
- learning rate
- tokens processed
- skipped optimizer steps

NaN 또는 Inf 발생 시:

1. 현재 checkpoint 보존
2. 직전 안정 checkpoint로 복원
3. 학습률과 scaler 상태 확인
4. 문제 batch 또는 데이터 기록

---

## 18. Checkpoint 스펙

### 18.1 저장 내용

```text
model state
optimizer state
scheduler state
gradient scaler state
global step
tokens processed
epoch 또는 data position
random number generator states
config
tokenizer identifier
data manifest identifier
wandb run id
git commit hash
```

### 18.2 저장 정책

```text
Periodic checkpoint
Evaluation checkpoint
Best validation checkpoint
Final checkpoint
```

권장:

- 평가 주기마다 checkpoint 저장
- 최신 N개 periodic checkpoint 유지
- best checkpoint 별도 유지

### 18.3 Resume

Resume 시 다음이 이어져야 한다.

- model weights
- optimizer momentum
- learning rate schedule
- gradient scaler
- global step
- tokens processed
- data sampling state
- W&B run

가능한 한 동일한 다음 batch부터 이어가되, streaming 제약으로 정확한 위치 재현이 어려운 경우 그 사실을 기록한다.

---

## 19. Continued Pretraining 스펙

### 19.1 입력 모델

동일한 continued 레시피(§19.2/§19.3)를 여러 크기의 Base에 적용한다. 모델 dims 외에는 데이터·토큰 버짓·LR이 크기와 무관하게 같다.

```text
Kkoma-125M-Base   → Kkoma-125M-Ko-Base   (configs/continued_pretraining/ko_125m.yaml)
Kkoma-350M-Base   → Kkoma-350M-Ko-Base   (ko_350m.yaml)
Kkoma-800M-Base   → Kkoma-800M-Ko-Base   (ko_800m.yaml)
Kkoma-1B-Base     → Kkoma-1B-Ko-Base     (ko_1b.yaml)
Kkoma-1.3B-Base   → Kkoma-1.3B-Ko-Base   (ko_1.3b.yaml)
```

### 19.2 데이터 mixture

```text
FineWeb2 Korean: 70%
Base English replay: 30%
```

기본 token budget:

```text
2B tokens
```

세부 구성:

```text
Korean: 1.4B tokens
English replay: 0.6B tokens
```

### 19.3 학습 설정

```text
Initial peak LR: 5e-5
Optimizer: AdamW
Betas: (0.9, 0.95)
Weight decay: 0.1
Gradient clipping: 1.0
Warmup: 1~2%
Scheduler: Cosine decay
Context length: 1,024
```

Base 모델의 optimizer state를 이어갈지 새로 시작할지는 실험 config로 명시한다. Kkoma v1 기본안은 **새 optimizer state로 시작**한다.

### 19.4 Forgetting 대응

기본 70:30 mixture에서 영어 성능 저하가 큰 경우에만 다음을 검토한다.

```text
Korean 50% / English 50%
Korean 70% / English 30%
Korean 90% / English 10%
```

v1에서는 광범위한 grid search를 수행하지 않는다.

---

## 20. 추론 및 KV Cache 스펙

### 20.1 기본 generation

지원 옵션:

```text
max_new_tokens
temperature
top_k
seed
eos_token_id
```

선택 확장:

```text
top_p
repetition penalty
batch generation
```

### 20.2 KV Cache

각 layer에서 저장:

```text
K cache: [batch, n_kv_heads, cached_seq, head_dim]
V cache: [batch, n_kv_heads, cached_seq, head_dim]
```

요구사항:

- cache 사용/미사용 generation 결과가 greedy decoding에서 동일해야 함
- position offset이 RoPE에 올바르게 반영되어야 함
- cache 길이가 context limit을 넘을 때 명시적 오류 또는 정책 적용
- MHA와 GQA 모두 지원

### 20.3 Stop condition

Base generation:

```text
<|eos|>
```

Kkoma-Chat 예정 generation:

```text
<|turn_end|>
```

Kkoma-LLM에서는 chat generation을 구현 필수로 두지 않는다.

---

## 21. 평가 스펙

### 21.1 Language Modeling

필수:

- 전체 validation loss
- 영어 validation loss
- 한국어 validation loss
- perplexity
- token-weighted 평균

### 21.2 효율

- tokens per second
- step time
- peak allocated GPU memory
- peak reserved GPU memory
- total training time
- processed tokens
- parameter count

### 21.3 Downstream

Base 모델에 적합한 zero-shot 또는 few-shot 평가를 사용한다.

초기 후보:

- LAMBADA
- HellaSwag
- PIQA
- ARC-Easy
- WinoGrande
- 한국어 상식 또는 언어 이해 benchmark 일부

평가 harness 연동 시 tokenizer와 prompt format을 고정한다.

작은 모델의 benchmark 절대 점수보다 모델 크기 및 단계별 변화에 초점을 둔다.

### 21.4 Generation Samples

고정 prompt set을 유지한다.

영어:

```text
The meaning of life is
Artificial intelligence can
In the future, small language models
```

한국어:

```text
인공지능이란
대한민국의 수도는
작은 언어 모델을 직접 학습하면
```

모든 checkpoint에서 같은 prompt와 sampling seed로 생성한다.

---

## 22. 테스트 요구사항

### 22.1 Unit tests

필수:

- tokenizer encode/decode round trip
- special token 단일 ID 변환
- vocabulary size 확인
- RoPE shape 및 offset
- RMSNorm 수치
- SwiGLU shape
- GQA head mapping
- causal masking
- weight tying
- checkpoint save/load
- KV cache equivalence

### 22.2 Integration tests

- TinyStories 소규모 overfit
- single-GPU training
- multi-GPU DDP training
- gradient accumulation equivalence
- interrupted run resume
- deterministic eval
- Base checkpoint에서 continued pretraining 시작

### 22.3 Overfit test

아주 작은 batch를 반복 학습하여 loss가 충분히 낮아지는지 확인한다. 새로운 아키텍처 모듈을 추가할 때마다 overfit test를 먼저 통과해야 한다.

---

## 23. 로깅 스펙

Weights & Biases 기본 필드:

```text
train/loss
val/loss
val/loss_en
val/loss_ko
train/perplexity
val/perplexity
train/learning_rate
train/grad_norm
train/tokens_per_second
train/step_time
system/gpu_memory
progress/tokens_processed
progress/global_step
```

W&B 프로젝트는 학습 단계별로 분리한다. 같은 단계의 모든 크기는 하나의 프로젝트를 공유하고, 크기·구성
구분은 run name으로 한다. `project.name`(RunConfig 최상단)은 항상 `kkoma-llm`으로 두고, wandb 프로젝트는
`logging.project`로만 지정한다.

```text
kkoma-llm-pt      Base pretraining        (configs/pretraining/base_*.yaml)
kkoma-llm-cpt     Continued pretraining   (configs/continued_pretraining/ko_*.yaml)
kkoma-llm-core    Architecture 실험        (configs/architecture/core_*.yaml)
```

Run name 예:

```text
core-125m-rope-seed42
base-125m-tpp10
base-350m-tpp10
base-800m-tpp10
base-1b-tpp10
base-1.3b-tpp10
ko-125m-cpt-2b
ko-350m-cpt-2b
ko-800m-cpt-2b
ko-1b-cpt-2b
ko-1.3b-cpt-2b
```

Resume 시 동일한 W&B run ID를 재사용한다.

---

## 24. Config 스펙

모든 run config는 다음 범주를 포함해야 한다.

```text
project
model
tokenizer
data
training
optimizer
scheduler
distributed
evaluation
logging
checkpoint
```

예:

```yaml
project:
  name: kkoma-llm
  run_name: base-125m-tpp10
  seed: 42

model:
  vocab_size: 32768
  context_length: 1024
  n_layer: 12
  d_model: 768
  n_head: 12
  n_kv_head: 3
  norm: rmsnorm
  positional_encoding: rope
  activation: swiglu
  tie_word_embeddings: true
  bias: false
  dropout: 0.0

training:
  precision: fp16
  global_batch_tokens: 262144
  max_tokens: 2500000000
  grad_clip: 1.0

optimizer:
  name: adamw
  learning_rate: 0.0006
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

scheduler:
  name: cosine
  warmup_ratio: 0.02
  min_lr_ratio: 0.1

logging:
  backend: wandb
  project: kkoma-llm-pt   # 단계별 프로젝트: pretraining=-pt, continued=-cpt, architecture=-core
```

---

## 25. 재현성 스펙

각 실험은 다음 정보를 반드시 남긴다.

- Git commit hash
- full config
- tokenizer hash
- data manifest
- dataset revision
- random seeds
- package versions
- GPU 정보
- 시작 및 종료 시각
- total tokens
- checkpoint path
- evaluation result
- W&B run URL 또는 ID

데이터 manifest에는 다음을 포함한다.

```text
dataset name
dataset revision
subset
sampling ratio
sampling seed
document count
token count
train/validation split method
output shard checksum
```

---

## 26. 모델 및 파일 명명 규칙

공개 모델:

```text
Kkoma-125M-Base
Kkoma-350M-Base
Kkoma-800M-Base
Kkoma-1B-Base
Kkoma-1.3B-Base
Kkoma-125M-Ko-Base
Kkoma-350M-Ko-Base
Kkoma-800M-Ko-Base
Kkoma-1B-Ko-Base
Kkoma-1.3B-Ko-Base
```

내부 run:

```text
kkoma-core-125m-rope
kkoma-base-125m-tpp10
kkoma-base-350m-tpp10
kkoma-base-800m-tpp10
kkoma-base-1b-tpp10
kkoma-base-1.3b-tpp10
kkoma-ko-125m-cpt2b
kkoma-ko-350m-cpt2b
kkoma-ko-800m-cpt2b
kkoma-ko-1b-cpt2b
kkoma-ko-1.3b-cpt2b
```

Checkpoint:

```text
step_00001000.pt
step_00002000.pt
best_val.pt
final.pt
```

---

## 27. 단계별 완료 조건

### Phase 0 완료

- 최종 vocabulary가 32,768이다.
- 모든 특수 토큰이 단일 token으로 인코딩된다.
- 영어·한국어 encode/decode round trip이 정상 동작한다.
- GPT-2 대비 한국어 token 효율이 개선된다.
- tokenizer artifact와 data manifest가 저장된다.

### Phase 1 완료

- 네 가지 핵심 구조 비교가 동일 조건에서 완료된다.
- 모든 실험이 같은 tokenizer와 corpus를 사용한다.
- validation loss, 처리량, 메모리 결과가 정리된다.
- 최종 Kkoma Base 구조가 확정된다.
- KV cache generation 테스트가 통과한다.

### Phase 2 완료

- 125M, 350M, 800M, 1B, 1.3B 학습이 각각 목표 token budget까지 완료된다.
- 각 모델의 final 및 best checkpoint가 저장된다.
- scaling 및 compute efficiency 결과가 정리된다.
- Base generation sample과 downstream 평가가 생성된다.
- 중단 후 resume가 검증된다.

### Phase 3 완료

- 800M 모델이 2B-token continued pretraining을 완료한다.
- 한국어 validation loss와 benchmark가 개선된다.
- 영어 능력 변화가 측정된다.
- catastrophic forgetting 결과가 보고된다.
- Kkoma-Chat에서 로드 가능한 최종 checkpoint가 생성된다.

---

## 28. Kkoma-LLM v1 비기능 요구사항

- 핵심 모델 코드는 한 사람이 읽고 이해할 수 있는 수준으로 유지한다.
- 지나친 추상화와 프레임워크 의존을 피한다.
- 모든 핵심 모듈은 config로 교체 가능해야 한다.
- 실패 시 원인을 추적할 수 있도록 상태를 충분히 기록한다.
- 데이터와 모델 결과가 재현 가능해야 한다.
- V100 8장 환경에서 실제 학습 가능한 구성을 우선한다.
- 성능 최적화보다 정확성과 이해 가능성을 먼저 확보한다.

---

## 29. v1 이후 확장 지점

Kkoma-LLM v1 완료 후 다음 프로젝트가 동일 tokenizer와 checkpoint 형식을 사용한다.

```text
Kkoma-Chat
- supervised fine-tuning
- response-only loss
- chat template
- DPO
- dialogue evaluation
- PPO, GRPO, RLHF는 제외

Kkoma-Agent
- tool use
- function calling
- planning
- multi-step execution
- action-result feedback
- agent runtime and task evaluation
- 정확한 학습 방식은 Kkoma-Agent 설계 단계에서 결정
```

예약 특수 토큰은 Kkoma-Agent에서 실제 프로토콜이 확정될 때 의미를 부여한다. Kkoma-LLM 단계에서는 tool 및 Agent execution protocol을 확정하지 않는다.
