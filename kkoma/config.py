"""Configuration objects for Kkoma-LLM.

All run configs are expressed as nested dataclasses that mirror the YAML
schema described in docs/spec.md section 24. The config is the
single source of truth: every core module is replaceable through config so
that architecture experiments only change one component at a time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture configuration for a decoder-only Transformer.

    Every structural choice that the Architecture Study compares lives here so
    that a single field flip (e.g. ``norm`` or ``positional_encoding``) produces
    a different model while everything else stays fixed.
    """

    vocab_size: int = 32768
    context_length: int = 1024
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    # ``None`` -> defaults to ``n_head`` (i.e. plain multi-head attention).
    n_kv_head: Optional[int] = None
    # ``None`` -> derived as ``d_model // n_head``.
    head_dim: Optional[int] = None
    # ``None`` -> derived from the activation (4*d for gelu, ~8/3*d for swiglu).
    d_ff: Optional[int] = None

    # Swappable components (Architecture Study knobs).
    norm: str = "rmsnorm"  # {"rmsnorm", "layernorm"}
    positional_encoding: str = "rope"  # {"rope", "learned"}
    activation: str = "swiglu"  # {"swiglu", "gelu"}

    # Numerics / structural flags.
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    bias: bool = False
    dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.head_dim is None:
            assert self.d_model % self.n_head == 0, "d_model must divide n_head"
            self.head_dim = self.d_model // self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by "
                f"n_kv_head ({self.n_kv_head})"
            )
        if self.d_ff is None:
            self.d_ff = default_ffn_dim(self.d_model, self.activation)

    @property
    def n_query_groups(self) -> int:
        return self.n_head // self.n_kv_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim


def default_ffn_dim(d_model: int, activation: str, multiple_of: int = 128) -> int:
    """Pick an FFN inner dimension that keeps block params roughly matched.

    GELU MLP uses ``4 * d_model``. SwiGLU adds a third projection, so the spec
    recommends ``~8/3 * d_model`` and rounding to a multiple of 64/128 for
    tensor-core efficiency.
    """

    if activation == "gelu":
        target = 4 * d_model
    elif activation == "swiglu":
        target = int(8 * d_model / 3)
    else:
        raise ValueError(f"unknown activation: {activation}")
    return _round_to_multiple(target, multiple_of)


def _round_to_multiple(value: int, multiple_of: int) -> int:
    return multiple_of * ((value + multiple_of - 1) // multiple_of)


# ---------------------------------------------------------------------------
# Tokenizer / data / training / etc.
# ---------------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    # bos/eos insertion and vocab size are not knobs here: packing always
    # wraps documents as <|bos|> ... <|eos|> (spec 14.3) and the vocabulary
    # size lives in model.vocab_size (checked against the loaded tokenizer).
    path: str = "artifacts/tokenizer"


@dataclass
class DataSource:
    name: str
    weight: float = 1.0
    split: str = "train"
    path: Optional[str] = None  # local jsonl/parquet shard glob
    revision: Optional[str] = None
    subset: Optional[str] = None
    text_key: str = "text"


@dataclass
class DataConfig:
    sources: list[DataSource] = field(default_factory=list)
    sampling_seed: int = 42
    min_doc_chars: int = 16
    # What the source weights are a ratio of. "token" (default) makes 95/5 and
    # 70/30 mean what they say, which is what a language-mixture target means;
    # "document" weights by draw frequency instead. They diverge sharply when
    # document lengths differ — see kkoma/data/mixture.py.
    mixture_weighting: str = "token"
    # Validation holdout produced from the same pipeline.
    val_sources: list[DataSource] = field(default_factory=list)


@dataclass
class TrainingConfig:
    precision: str = "fp16"  # {"fp16", "bf16", "fp32"}
    global_batch_tokens: int = 262_144
    micro_batch_size: int = 4
    grad_accum_steps: Optional[int] = None  # derived if None
    max_tokens: int = 2_500_000_000
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 1000
    eval_tokens: int = 10_000_000
    # Downstream benchmarks run on their own cadence, separate from validation
    # loss so it can be tuned without disturbing the loss curve. 0 disables them.
    downstream_interval: int = 1000
    # On resume, skip the already-consumed prefix of the deterministic data
    # stream so training continues with the exact next batch (spec 18.3).
    # Re-tokenizes the skipped prefix once, which can take a while for large
    # corpora; set false to restart the stream instead (divergence recorded).
    resume_fastforward: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    find_unused_parameters: bool = False


@dataclass
class DownstreamTask:
    """One frozen benchmark set built by scripts/prepare_downstream_data.py."""

    name: str
    path: str
    language: str = "en"  # groups the en_avg / ko_avg aggregates
    enabled: bool = True
    # Scored on the in-training cadence (downstream_interval). Heavier tasks can
    # be reserved for the post-training run (scripts/evaluate.py) by setting this
    # false: the trainer then skips them, but the final evaluation still scores
    # every enabled task. Keeps the training loop's benchmark pass light.
    during_training: bool = True


@dataclass
class EvaluationConfig:
    # Validation cadence is training.eval_interval; this section holds the
    # fixed generation-sample settings (spec 21.4).
    generation_prompts_en: list[str] = field(
        default_factory=lambda: [
            "The meaning of life is",
            "Artificial intelligence can",
            "In the future, small language models",
        ]
    )
    generation_prompts_ko: list[str] = field(
        default_factory=lambda: [
            "인공지능이란",
            "대한민국의 수도는",
            "작은 언어 모델을 직접 학습하면",
        ]
    )
    sampling_seed: int = 1234

    # ---- downstream benchmarks (spec 21.3) --------------------------------
    # Master switch: false skips them everywhere regardless of the task list.
    downstream_enabled: bool = True
    # Questions scored per forward pass. Sequences are right-padded, which is
    # safe here only because attention is causal: a real token at position i
    # attends to positions <= i, so trailing pad can never reach it. Left
    # padding would shift RoPE positions and silently corrupt the scores.
    downstream_batch_size: int = 16
    downstream_tasks: list[DownstreamTask] = field(default_factory=list)


@dataclass
class LoggingConfig:
    backend: str = "wandb"  # {"wandb", "none"}
    project: str = "kkoma-llm"
    wandb_run_id: Optional[str] = None


@dataclass
class CheckpointConfig:
    dir: str = "artifacts/checkpoints"
    save_interval: int = 1000
    keep_last: int = 3
    save_best: bool = True


@dataclass
class ProjectConfig:
    name: str = "kkoma-llm"
    run_name: str = "base-125m"
    seed: int = 42
    # Independent seeds tracked per spec section 16.3.
    data_seed: int = 42
    init_seed: int = 42
    sampling_seed: int = 1234


@dataclass
class RunConfig:
    """Top-level config aggregating every category from spec section 24."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ---- serialization helpers -------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        cfg = _from_dict(cls, data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject values that would otherwise change the experiment silently.

        These fields are all dispatched on by string comparison, so a typo does
        not raise — it falls through to a different branch. ``precision:
        float16`` (instead of ``fp16``) leaves ``use_amp`` False and trains the
        whole run in FP32; ``backend: wanb`` logs to the console for hours. An
        architecture study that changes one component at a time cannot absorb
        that, so the mismatch has to surface at config-load time.
        """

        allowed = {
            "training.precision": (self.training.precision, {"fp16", "bf16", "fp32"}),
            "model.norm": (self.model.norm, {"rmsnorm", "layernorm"}),
            "model.positional_encoding": (
                self.model.positional_encoding, {"rope", "learned"},
            ),
            "model.activation": (self.model.activation, {"swiglu", "gelu"}),
            "optimizer.name": (self.optimizer.name, {"adamw"}),
            "scheduler.name": (self.scheduler.name, {"cosine"}),
            "logging.backend": (self.logging.backend, {"wandb", "none"}),
            "data.mixture_weighting": (self.data.mixture_weighting, {"token", "document"}),
        }
        for key, (value, options) in allowed.items():
            if value not in options:
                raise ValueError(
                    f"{key}={value!r} is not one of {sorted(options)}"
                )
        for task in self.evaluation.downstream_tasks:
            if task.language not in {"en", "ko"}:
                raise ValueError(
                    f"downstream task {task.name!r} has language={task.language!r}; "
                    "expected 'en' or 'ko' (it drives the en_avg / ko_avg aggregates)"
                )

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml  # local import; PyYAML is an optional dependency

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def _from_dict(klass: type, data: Any) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    ``from __future__ import annotations`` stores field types as strings, so we
    resolve the real types with ``get_type_hints``. Handles ``list[DataSource]``
    style fields. Unknown keys are an error: a silently dropped typo (e.g.
    ``n_kv_heads``) would fall back to a default and quietly break the
    one-component-changed invariant the architecture study depends on.
    """

    if not is_dataclass(klass):
        return data
    if data is None:
        return klass()
    import typing

    known = {f.name for f in fields(klass)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {klass.__name__}: {sorted(unknown)} "
            f"(known: {sorted(known)})"
        )

    hints = typing.get_type_hints(klass)
    kwargs: dict[str, Any] = {}
    for f in fields(klass):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
    return klass(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    import typing

    # Nested dataclass
    if is_dataclass(hint) and isinstance(value, dict):
        return _from_dict(hint, value)
    # list[DataSource] (or any list of dataclasses)
    if isinstance(value, list):
        args = typing.get_args(hint)
        if args and is_dataclass(args[0]):
            elem_type = args[0]
            return [
                _from_dict(elem_type, v) if isinstance(v, dict) else v for v in value
            ]
        return value
    # Scalars: YAML 1.1 does not recognize unquoted scientific notation, so
    # `learning_rate: 6e-4` arrives as the string '6e-4' and would silently
    # travel all the way into the optimizer. Cast to the annotated type instead.
    # Optional[T] is unwrapped; bool is left alone (bool is a subclass of int,
    # and int("true") is not what anyone means).
    target = hint
    args = typing.get_args(hint)
    if args:  # Optional[T] / Union[T, None]
        non_none = [a for a in args if a is not type(None)]
        target = non_none[0] if len(non_none) == 1 else None
    if target in (int, float) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
        try:
            return target(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected {target.__name__}, got {value!r}") from None
    return value


# ---------------------------------------------------------------------------
# Derived quantities (single source of truth for train / scheduler wiring)
# ---------------------------------------------------------------------------


def resolve_grad_accum(config: "RunConfig", world_size: int = 1) -> int:
    """Gradient-accumulation steps used by an optimizer step.

    If ``grad_accum_steps`` is set it wins; otherwise it is derived from
    ``global_batch_tokens``. Used by both the Trainer and the training-script
    wiring so the scheduler horizon always matches the steps actually run.
    """

    t = config.training
    if t.grad_accum_steps:
        return t.grad_accum_steps
    denom = t.micro_batch_size * config.model.context_length * max(world_size, 1)
    return max(1, t.global_batch_tokens // max(denom, 1))


def tokens_per_optimizer_step(config: "RunConfig", world_size: int = 1) -> int:
    return (
        config.training.micro_batch_size
        * config.model.context_length
        * resolve_grad_accum(config, world_size)
        * max(world_size, 1)
    )


__all__ = [
    "ModelConfig",
    "TokenizerConfig",
    "DataSource",
    "DataConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DistributedConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "ProjectConfig",
    "RunConfig",
    "default_ffn_dim",
    "resolve_grad_accum",
    "tokens_per_optimizer_step",
]
