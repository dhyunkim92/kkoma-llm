# Kkoma-LLM Design Rationale

## 1. 프로젝트 개요

**Kkoma-LLM**은 현대적인 소형 decoder-only 언어 모델을 토크나이저와 아키텍처부터 직접 설계하고, Base pretraining과 한국어 continued pretraining까지 수행하는 학습·실험 중심 프로젝트다.

기존 공개 모델을 가져와 fine-tuning하는 것이 아니라 다음 전체 과정을 직접 경험하는 것을 목표로 한다.

1. 영어와 한국어를 함께 처리할 수 있는 전용 토크나이저를 학습한다.
2. 현대 LLM의 핵심 구성 요소를 직접 구현하고 비교한다.
3. 확정한 구조로 여러 크기의 Base 모델을 처음부터 학습한다.
4. 영어 중심 Base 모델을 한국어 데이터에 추가 적응시킨다.
5. 이후 post-training과 agent 개발로 이어질 공통 기반을 마련한다.

Kkoma-LLM의 일차 목표는 최고 성능이나 SOTA 달성이 아니다.

> 토크나이저, 아키텍처, 데이터, 모델 크기, 학습 레시피가 언어 모델의 성능과 효율에 어떤 영향을 미치는지 직접 구현하고 설명할 수 있는 수준으로 이해한다.

---

## 2. 이름과 프로젝트 정체성

**Kkoma**는 한국어의 ‘꼬마’에서 가져온 이름이다.

작고 아직 완전하지 않지만, 직접 만들고 키워 나가는 소형 언어 모델이라는 프로젝트의 성격을 나타낸다. Kkoma 프로젝트는 대규모 상용 모델을 단순히 축소 모방하는 데 그치지 않고, 제한된 계산 자원 안에서 LLM 개발의 전체 흐름을 끝까지 완주하는 것을 중요하게 본다.

프로젝트의 핵심 정체성은 다음과 같다.

- 작지만 완결된 모델 개발 과정
- 단순하지만 현대적인 구현
- 통제된 비교와 재현 가능한 실험
- 영어 Base에서 한국어 모델로 이어지는 확장
- Base model부터 Chat과 Agent까지 이어지는 단계적 성장
- 각 단계의 목적을 분리한 프로젝트 구성

---

## 3. 전체 Kkoma 프로젝트 구성

Kkoma는 하나의 모델이 아니라, LLM을 처음부터 만들고 단계적으로 능력을 확장하는 프로젝트 계열이다.

```text
Kkoma-LLM
    ↓
Kkoma-Chat
    ↓
Kkoma-Agent
```

### 3.1 Kkoma-LLM: Preparation, Pre-training, Mid-training

토크나이저와 모델 구조를 준비하고, Base pretraining과 한국어 mid-training을 담당한다.

주요 범위:

- Preparation: bilingual tokenizer, 데이터 파이프라인, architecture study
- Pre-training: 범용 Base 모델의 next-token prediction 학습
- Mid-training: 한국어 continued pretraining
- 모델 규모별 scaling 분석
- 공통 checkpoint 및 inference 기반 마련

대표 산출물:

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

### 3.2 Kkoma-Chat: Post-training

Kkoma Base 모델을 사용자의 지시를 따르고 대화할 수 있는 모델로 변환한다. Kkoma-Chat v1의 post-training은 **SFT 이후 DPO를 적용하는 흐름**으로 제한한다.

주요 범위:

- supervised fine-tuning
- chat formatting
- response-only loss
- DPO 기반 preference optimization
- 대화 및 instruction-following 평가
- PPO, GRPO, RLHF 등 RL 기반 post-training은 v1에서 제외

대표 산출물:

```text
Kkoma-800M-Instruct
Kkoma-800M-Chat
```

### 3.3 Kkoma-Agent: Agent System

대화 모델을 외부 환경과 상호작용하며 실제 작업을 수행할 수 있는 Agent로 확장한다. 정확한 agent training recipe는 Kkoma-Agent 설계 단계에서 별도로 결정하며, 현재 로드맵에서는 RL 기반 학습을 전제하지 않는다.

주요 범위:

- tool use
- function calling
- planning
- multi-step execution
- state and context management
- action-result feedback loop
- agent trajectory training
- task completion evaluation

Agent는 단순히 특수 토큰을 추가하는 단계가 아니라, 모델의 판단과 외부 실행을 연결하는 별도 시스템 프로젝트로 다룬다.

---

## 4. 학습 단계 분류

Kkoma 프로젝트는 일반적인 LLM 개발 용어에 맞춰 다음처럼 구분한다.

```text
Preparation
→ Pre-training
→ Mid-training
→ Post-training
→ Agent System
```

| 분류 | Kkoma 단계 | 핵심 학습 방식 |
|---|---|---|
| Preparation | Tokenizer, Data, Architecture Study | 학습 기반 준비 및 구조 실험 |
| Pre-training | Kkoma Base | 일반 원문 전체에 대한 causal next-token prediction |
| Mid-training | Kkoma-Ko Base | 한국어 중심 원문에 대한 continued next-token prediction |
| Post-training | Kkoma-Chat | SFT 후 DPO |
| Agent System | Kkoma-Agent | Tool, planning, execution runtime; 학습 방식은 추후 결정 |

Pre-training과 Mid-training은 모두 일반 원문에 대해 next-token prediction을 수행한다. Mid-training은 objective가 아니라 데이터 분포, learning rate, token budget과 replay 전략이 달라지는 단계다.

Kkoma-Chat v1은 다음 흐름으로 고정한다.

```text
Kkoma-800M-Ko-Base
→ SFT
→ Kkoma-800M-Instruct
→ DPO
→ Kkoma-800M-Chat
```

PPO, GRPO, RLHF 및 기타 online RL 기반 post-training은 현재 범위에서 제외한다.

---

## 5. Kkoma-LLM 프로젝트 범위

Kkoma-LLM은 전체 Kkoma 계열 중 토크나이저와 pretraining 계열만 담당한다.

```text
Phase 0. Tokenizer Preparation
        ↓
Phase 1. Architecture Study
        ↓
Phase 2. Base Pretraining
        ↓
Phase 3. Korean Continued Pretraining
```

Instruction tuning, preference alignment, tool use 및 Agent 개발은 Kkoma-LLM의 직접 범위에 포함하지 않는다.

---

## 6. 핵심 설계 원칙

### 5.1 직접 구현할 것과 재사용할 것을 구분한다

Kkoma-LLM은 모든 저수준 알고리즘을 처음부터 다시 만드는 프로젝트가 아니다.

학습 가치가 높은 모델 구조와 학습 로직은 직접 구현하되, 이미 안정적으로 제공되는 저수준 기능은 검증된 라이브러리를 활용한다.

예:

- 토크나이저 알고리즘 구현은 검증된 라이브러리를 사용한다.
- Kkoma 전용 vocabulary와 merge rules는 직접 학습한다.
- 모델 구조와 학습 루프는 직접 구현한다.
- CUDA kernel, 분산 통신, mixed precision은 기존 프레임워크를 활용한다.

> 이해가 필요한 핵심 구조는 직접 만들고, 핵심 목표와 무관한 저수준 최적화는 재사용한다.

### 5.2 단순하지만 현대적인 구조를 지향한다

Kkoma-LLM은 nanoGPT 수준의 읽기 쉬운 코드 구조를 유지하면서, 최근 decoder-only LLM에서 널리 사용되는 구성 요소를 단계적으로 적용한다.

처음부터 복잡한 구조를 모두 넣지 않는다. 기본 GPT형 모델에서 시작해 위치 인코딩, normalization, feed-forward network, attention 구조를 하나씩 교체하고 비교한 뒤 최종 구조를 확정한다.

### 5.3 중요한 차이만 통제된 조건에서 비교한다

Architecture Study에서는 구조 효과를 명확히 보기 위해 한 번에 하나의 구성 요소만 변경한다.

모든 공식 구조 실험은 다음 조건을 공유한다.

- 같은 모델 규모
- 같은 토크나이저
- 같은 학습 데이터
- 같은 token budget
- 같은 optimizer와 scheduler
- 같은 random seed와 평가 절차

실험 수를 무작정 늘리기보다, 이후 Base 모델의 설계 결정에 직접 영향을 주는 핵심 차이에 집중한다.

### 5.4 소형 모델의 파라미터와 계산 자원을 효율적으로 사용한다

Kkoma는 125M~1.3B 규모의 모델을 중심으로 한다.

대규모 모델에서 무시할 수 있는 embedding, vocabulary, cache, batch 구성의 차이가 소형 모델에서는 큰 영향을 줄 수 있다. 따라서 Kkoma는 모델 크기와 하드웨어 환경에 맞는 파라미터 효율, 메모리 효율, 학습 안정성을 우선한다.

최종 선택은 validation loss뿐 아니라 다음을 함께 고려한다.

- 파라미터 수
- 학습 처리량
- GPU 메모리
- 추론 효율
- 구현 복잡도
- 후속 확장 가능성

### 5.5 토크나이저는 전체 모델 계열이 공유하는 기반으로 설계한다

Kkoma tokenizer는 Kkoma-LLM에서만 사용하는 일회성 구성 요소가 아니다.

동일한 tokenizer를 다음 단계까지 유지하는 것을 원칙으로 한다.

```text
Kkoma-LLM
→ Kkoma-Chat
→ Kkoma-Agent
```

따라서 Base pretraining에 필요한 일반 텍스트뿐 아니라 이후 대화 및 확장 가능성을 고려해 설계한다.

구체적인 vocabulary 크기, 특수 토큰 목록, 예약 토큰 수, chat formatting 규칙은 구현 세부사항이므로 `spec.md`에서 정의한다.

### 5.6 현재 단계와 미래 Agent 기능의 경계를 유지한다

Kkoma-LLM 단계에서 tool use와 Agent protocol을 미리 확정하지 않는다.

미래의 Agent 기능을 예상해 Base tokenizer와 model architecture에 구체적인 프로토콜을 과도하게 반영하면 현재 프로젝트의 범위가 불필요하게 복잡해진다.

따라서 다음 원칙을 따른다.

- 범용적으로 필요한 확장 여지만 확보한다.
- tool use와 execution protocol은 Kkoma-Agent에서 다룬다.
- Kkoma-LLM은 안정적인 Base와 공통 인터페이스 제공에 집중한다.

### 5.7 학습 기능과 추론 기능을 구분한다

Kkoma-LLM은 pretraining에 필요한 학습 기능과 checkpoint를 검증하고 활용하기 위한 추론 기능을 분리해 구현한다.

학습 측면:

- causal language modeling
- mixed precision
- distributed training
- checkpoint save/resume
- validation

추론 측면:

- autoregressive generation
- sampling
- KV cache
- generation efficiency measurement

---

## 7. 단계별 목표

### Phase 0. Tokenizer Preparation

영어와 한국어를 함께 처리할 수 있는 Kkoma 전용 tokenizer를 준비한다.

목표:

- 영어와 한국어 모두에서 합리적인 token 효율 확보
- 소형 모델의 embedding 부담 통제
- 모든 Kkoma 프로젝트에서 공통 사용
- Base 학습 이후 tokenizer를 교체하지 않고 확장 가능
- 재현 가능한 학습 및 평가 절차 확립

세부 vocabulary, 특수 토큰, 학습 데이터 비율과 파일 형식은 `spec.md`에서 정의한다.

### Phase 1. Architecture Study: Kkoma Core

현대 decoder-only LLM의 핵심 구성 요소를 직접 구현하고 비교한다.

목표:

- 각 구조의 동작 원리 이해
- 학습 안정성과 loss 차이 확인
- 파라미터 및 메모리 효율 분석
- 학습과 추론 속도 비교
- Base pretraining에 사용할 최종 구조 확정

### Phase 2. Base Pretraining: Kkoma Base

확정된 구조로 여러 크기의 Base 모델을 처음부터 학습한다.

목표:

- 125M, 350M, 800M, 1B, 1.3B 모델의 end-to-end pretraining 완주
- 모델 크기에 따른 scaling behavior 분석
- 계산량, 데이터량, 성능 사이의 관계 파악
- 후속 언어 적응과 post-training을 위한 checkpoint 확보
- 재현 가능한 학습·평가·resume 파이프라인 완성

### Phase 3. Korean Continued Pretraining: Kkoma-Ko Base

영어 중심 Base 모델을 한국어 데이터에 추가 적응시킨다.

목표:

- 한국어 언어 모델링 능력 향상
- 기존 영어 능력 유지
- catastrophic forgetting 분석
- continued pretraining의 효율과 한계 확인
- Kkoma-Chat의 기반이 될 bilingual checkpoint 생성

---

## 8. 데이터 전략

Kkoma-LLM은 데이터셋 종류를 무작정 늘리지 않는다.

데이터 선택의 원칙은 다음과 같다.

- 공개성과 재현 가능성
- 충분한 규모
- 영어와 한국어 지원
- 단계별 목적에 맞는 품질
- 라이선스와 출처 추적 가능성
- 동일한 데이터 파이프라인의 재사용 가능성

데이터는 크게 다음 용도로 나눈다.

```text
Tokenizer 학습 데이터
Architecture smoke-test 데이터
Architecture 비교용 고정 corpus
Base pretraining corpus
Korean continued pretraining corpus
Validation 및 downstream evaluation data
```

Base pretraining corpus는 영어 전용이 아니라 영어와 한국어를 함께 포함한다. 영어가 대부분을 차지하지만 한국어도 일정 비율 섞여, 이후 한국어 continued pretraining이 완전히 새로운 언어에서 시작하지 않도록 한다.

구체적인 데이터셋, 비율, token budget, split 방식은 `spec.md`에서 정의한다.

---

## 9. Kkoma-LLM v1에서 제외하는 항목

프로젝트를 실제로 완성하기 위해 다음 항목은 Kkoma-LLM v1 범위에서 제외한다.

- supervised fine-tuning과 DPO는 Kkoma-Chat에서 수행
- PPO, GRPO, RLHF 등 RL 기반 post-training
- tool-use training
- function calling
- Agent training
- long-context pretraining
- Mixture-of-Experts
- hybrid sequence architecture
- 대규모 synthetic data pipeline

SFT와 DPO는 Kkoma-Chat에서 다루고, tool use와 Agent runtime은 Kkoma-Agent에서 다룬다. RL 기반 post-training은 현재 전체 로드맵에서 제외한다.

---

## 10. 최종 산출물

### 공통 기반

- Kkoma tokenizer
- 공통 model configuration
- checkpoint 형식
- generation interface
- evaluation interface

### 모델

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

### 코드

- configurable decoder-only Transformer
- tokenizer training pipeline
- architecture comparison pipeline
- pretraining pipeline
- continued pretraining pipeline
- distributed training
- checkpoint resume
- language-model evaluation
- autoregressive generation 및 KV cache

### 분석 결과

- tokenizer 효율 분석
- architecture ablation
- 모델 규모별 scaling 비교
- 학습 처리량 및 메모리 분석
- 한국어 적응 효과
- catastrophic forgetting 분석
- 재현 가능한 config와 로그

---

## 11. 성공 기준

Kkoma-LLM v1은 다음 조건을 만족하면 완료된 것으로 본다.

1. 영어·한국어 tokenizer를 재현 가능하게 학습할 수 있다.
2. 핵심 아키텍처 비교를 동일한 조건에서 수행할 수 있다.
3. 최종 Base 구조를 실험 결과에 근거해 선택할 수 있다.
4. 125M, 350M, 800M, 1B, 1.3B Base 모델의 학습을 완료한다.
5. checkpoint 중단 및 재개가 안정적으로 동작한다.
6. 모델 크기에 따른 성능과 효율 변화를 비교할 수 있다.
7. 최종 Base 모델을 한국어에 continued pretraining한다.
8. 한국어 성능 향상과 영어 능력 변화를 측정할 수 있다.
9. 동일 tokenizer와 checkpoint가 Kkoma-Chat 이후 단계로 연결된다.
10. 주요 실험의 config, seed, 데이터 버전, 코드 버전과 로그가 기록된다.

---

## 12. 최종 설계 철학

Kkoma-LLM은 작은 checkpoint 하나를 만드는 프로젝트가 아니다.

```text
Design the tokenizer
        ↓
Understand the architecture
        ↓
Build the model
        ↓
Pretrain from scratch
        ↓
Analyze scaling behavior
        ↓
Adapt the model to Korean
        ↓
Provide the foundation for Chat and Agents
```

> **Kkoma-LLM is a compact and reproducible language-model project built to understand the path from tokenizer design and architecture exploration to pretraining and Korean language adaptation, while providing the foundation for future chat and agent models.**
