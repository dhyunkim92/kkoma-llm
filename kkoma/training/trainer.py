"""The Kkoma training loop.

Single readable loop that covers: FP16 autocast + grad scaler, gradient
accumulation with DDP ``no_sync`` on non-final micro-steps, cosine LR schedule,
periodic + best-validation checkpointing, separate EN/KO validation loss, and
NaN/Inf stability handling. Resume restores all states.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from kkoma.config import RunConfig
from kkoma.training.checkpoint import (
    rotate_checkpoints,
    save_checkpoint,
)
from kkoma.training.distributed import DistInfo, all_reduce_sum, is_main_process
from kkoma.training.metrics import (
    Throughput,
    gpu_memory_stats,
    grad_global_norm,
    perplexity,
)


# ---------------------------------------------------------------------------
# Console helpers (tqdm-aware so prints don't break the progress bar)
# ---------------------------------------------------------------------------


def _console(msg: str) -> None:
    """Print a line without corrupting an active tqdm bar."""

    try:
        from tqdm import tqdm

        tqdm.write(msg)
    except Exception:
        print(msg)


class _NullBar:
    def update(self, n: int = 1) -> None: ...
    def set_postfix(self, **kwargs) -> None: ...
    def close(self) -> None: ...


def _make_step_bar(total: int, initial: int, desc: str):
    """A live per-step progress bar on the main process (no-op elsewhere)."""

    if not is_main_process():
        return _NullBar()
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullBar()
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit="step",
        dynamic_ncols=True,
        smoothing=0.1,
    )


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class Logger:
    """Minimal logger: W&B when configured, else stdout (rank 0 only)."""

    def __init__(self, config: RunConfig, dist: DistInfo):
        self.enabled = is_main_process()
        self.use_wandb = self.enabled and config.logging.backend == "wandb"
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=config.logging.project,
                    name=config.project.run_name,
                    id=config.logging.wandb_run_id,
                    resume="allow" if config.logging.wandb_run_id else None,
                    config=config.to_dict(),
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[logger] wandb unavailable ({exc}); falling back to stdout")
                self.use_wandb = False

    @property
    def run_id(self) -> Optional[str]:
        if self._wandb is not None and self._wandb.run is not None:
            return self._wandb.run.id
        return None

    def log(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            compact = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            _console(f"[step {step}] {compact}")

    def finish(self) -> None:
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    global_step: int = 0
    tokens_processed: int = 0
    best_val_loss: float = float("inf")
    skipped_steps: int = 0
    # True once the run was resumed: the streaming corpus restarts from its
    # fixed shuffled order, so the exact data position is not reproduced.
    # Recorded in checkpoints per spec section 18.3.
    data_position_diverged: bool = False


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        model: torch.nn.Module,
        raw_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: Iterable[dict],
        dist: DistInfo,
        device: torch.device,
        val_loaders: Optional[dict] = None,
        logger: Optional[Logger] = None,
        provenance: Optional[dict] = None,
        tokenizer=None,
        downstream_tasks: Optional[list] = None,
    ):
        self.config = config
        self.provenance = provenance or {}
        self.tokenizer = tokenizer
        self.downstream_tasks = downstream_tasks or []
        self.model = model  # possibly DDP-wrapped
        self.raw_model = raw_model  # underlying KkomaModel
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loaders = val_loaders or {}
        self.dist = dist
        self.device = device
        self.logger = logger or Logger(config, dist)
        self.state = TrainState()

        self.seq_len = config.model.context_length
        self.micro_bs = config.training.micro_batch_size
        self.grad_accum = self._resolve_grad_accum()
        self.tokens_per_step = (
            self.micro_bs * self.seq_len * self.grad_accum * dist.world_size
        )
        self.max_steps = max(1, config.training.max_tokens // self.tokens_per_step)

        self.precision = config.training.precision
        # FP16 is a CUDA feature; on CPU autocast-fp16 is unsupported and slow,
        # so fall back to fp32 (with a heads-up) rather than crawl or error.
        if device.type != "cuda" and self.precision == "fp16":
            if is_main_process():
                print("[trainer] CPU/non-CUDA device: forcing precision fp16 -> fp32")
            self.precision = "fp32"
        self.use_amp = self.precision in ("fp16", "bf16")
        self.amp_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        self.throughput = Throughput()
        # Best-so-far per validation language, used to flag a "best" checkpoint
        # whose aggregate improved only because one language carried the other.
        self._best_per_lang: dict[str, float] = {}

    def _resolve_grad_accum(self) -> int:
        from kkoma.config import resolve_grad_accum

        return resolve_grad_accum(self.config, self.dist.world_size)

    # ---- AMP-aware forward ------------------------------------------------
    def _forward_loss(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            _, loss = self.model(input_ids, labels=labels)
        return loss

    # ---- one optimizer step (grad_accum micro-batches) --------------------
    def _train_step(self, data_iter) -> Optional[dict]:
        # Fetch the whole accumulation window first, then agree across ranks
        # that everyone has a full one. Blocks are dealt round-robin, so ranks
        # exhaust the stream at different iterations; a rank leaving the loop
        # while the others are still in backward would mismatch the DDP
        # collectives and hang. The flag all-reduce keeps them in lockstep
        # (a partial window on the exhausted rank is discarded).
        batches = []
        for _ in range(self.grad_accum):
            try:
                batches.append(next(data_iter))
            except StopIteration:
                break
        have_full = torch.tensor(
            [1.0 if len(batches) == self.grad_accum else 0.0], device=self.device
        )
        if all_reduce_sum(have_full).item() < self.dist.world_size:
            return None  # some rank ran dry; every rank stops together

        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro, batch in enumerate(batches):
            is_last = micro == self.grad_accum - 1
            sync_ctx = (
                self.model.no_sync()
                if (self.dist.distributed and not is_last and hasattr(self.model, "no_sync"))
                else _nullcontext()
            )
            with sync_ctx:
                loss = self._forward_loss(batch) / self.grad_accum
                self.scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Unscale before clipping so the threshold is in real gradient units.
        self.scaler.unscale_(self.optimizer)
        grad_norm = grad_global_norm(self.raw_model.parameters())
        torch.nn.utils.clip_grad_norm_(
            self.raw_model.parameters(), self.config.training.grad_clip
        )

        # Non-finite loss on ANY rank: every rank must skip together. A local
        # decision would desynchronize global_step across ranks under bf16 or
        # fp32, where no grad scaler exists to coordinate the skip (fp16's
        # scaler handles its own overflow path below). unscale_ already
        # recorded found_inf, so update() still lets the FP16 scaler react.
        bad = torch.tensor(
            [0.0 if math.isfinite(accum_loss) else 1.0], device=self.device
        )
        if all_reduce_sum(bad).item() > 0:
            self.state.skipped_steps += 1
            self.scaler.update()
            if is_main_process():
                _console(
                    f"[trainer] non-finite loss at step {self.state.global_step}; "
                    "optimizer step skipped (spec 17.2: check LR/scaler if it repeats)"
                )
            return {"loss": float("nan"), "grad_norm": grad_norm, "skipped": 1}

        # scaler.step() internally skips the optimizer update if grads overflowed
        # (found_inf), and update() then halves the scale. Detect that via the
        # scale drop so a skipped step doesn't advance the LR schedule / counters.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() < scale_before:
            self.state.skipped_steps += 1
            return {"loss": accum_loss, "grad_norm": grad_norm, "skipped": 1}

        self.scheduler.step()
        self.state.global_step += 1
        self.state.tokens_processed += self.tokens_per_step
        self.throughput.update(self.tokens_per_step)
        return {
            "loss": accum_loss,
            "grad_norm": grad_norm,
            "skipped": 0,
        }

    # ---- main loop --------------------------------------------------------
    def _banner(self) -> None:
        if not is_main_process():
            return
        p = self.raw_model.num_parameters()
        m = self.config.model
        where = "wandb" if (self.logger.use_wandb) else "console"
        _console(
            "── Kkoma training ──────────────────────────────\n"
            f"  run        : {self.config.project.run_name}\n"
            f"  device     : {self.device} | precision {self.precision} | world_size {self.dist.world_size}\n"
            f"  model      : {p/1e6:.1f}M params | L{m.n_layer} d{m.d_model} "
            f"{m.activation}/{m.norm}/{m.positional_encoding} kv{m.n_kv_head}\n"
            f"  batch      : micro {self.micro_bs} × seq {self.seq_len} × accum {self.grad_accum} "
            f"× gpus {self.dist.world_size} = {self.tokens_per_step:,} tok/step\n"
            f"  schedule   : {self.max_steps:,} steps → {self.config.training.max_tokens/1e9:.2f}B tokens\n"
            f"  logging    : metrics → {where} | eval every {self.config.training.eval_interval} steps\n"
            "────────────────────────────────────────────────"
        )
        if self.device.type != "cuda":
            _console("  note: running on CPU: the first step can take a while; this is expected.")

    def train(self) -> TrainState:
        data_iter = iter(self.train_loader)
        self.model.train()
        self._banner()

        # Baseline before any update: downstream scores start at chance, and the
        # step-0 point is what shows a later rise is real rather than a
        # different measurement. Skipped on resume, where step > 0 already.
        if self.state.global_step == 0:
            self._evaluate_downstream(0)

        bar = _make_step_bar(self.max_steps, self.state.global_step, self.config.project.run_name)
        while self.state.global_step < self.max_steps:
            metrics = self._train_step(data_iter)
            if metrics is None:  # dataset exhausted
                if is_main_process():
                    _console("[trainer] data stream exhausted before reaching the token budget.")
                break
            step = self.state.global_step
            bar.update(1)

            if step % self.config.training.log_interval == 0:
                tps = self.throughput.rate()
                bar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    tok_s=f"{tps:,.0f}",
                )
                self._log_train(metrics, step, tokens_per_second=tps)
            if step % self.config.training.eval_interval == 0:
                self._evaluate(step)
            if (
                self.config.training.downstream_interval > 0
                and step % self.config.training.downstream_interval == 0
            ):
                self._evaluate_downstream(step)
            if step % self.config.checkpoint.save_interval == 0:
                self._save(step, tag=f"step_{step:08d}")

        bar.close()
        self._evaluate(self.state.global_step)
        self._evaluate_downstream(self.state.global_step)
        self._save(self.state.global_step, tag="final")
        self.logger.finish()
        return self.state

    # ---- logging / eval / checkpoint --------------------------------------
    def _log_train(self, metrics: dict, step: int, tokens_per_second: float | None = None) -> None:
        tps = tokens_per_second if tokens_per_second is not None else self.throughput.rate()
        payload = {
            "train/loss": metrics["loss"],
            "train/perplexity": perplexity(metrics["loss"]) if metrics["loss"] == metrics["loss"] else float("nan"),
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/grad_norm": metrics["grad_norm"],
            "train/tokens_per_second": tps,
            "train/scaler_scale": self.scaler.get_scale(),
            "progress/tokens_processed": self.state.tokens_processed,
            "progress/global_step": step,
            "train/skipped_steps": self.state.skipped_steps,
        }
        payload.update({f"system/{k}": v for k, v in gpu_memory_stats().items()})
        self.logger.log(payload, step)

    @torch.no_grad()
    def _evaluate(self, step: int) -> None:
        if not self.val_loaders:
            return
        # Avoid re-evaluating the same step (e.g. final eval landing on a
        # periodic eval boundary).
        if step == getattr(self, "_last_eval_step", None):
            return
        self._last_eval_step = step
        self.model.eval()
        results = {}
        for name, loader in self.val_loaders.items():
            loss = self._eval_loss(loader)
            results[f"val/loss_{name}" if name != "all" else "val/loss"] = loss
        if "val/loss" not in results and results:
            results["val/loss"] = sum(results.values()) / len(results)
        results["val/perplexity"] = perplexity(results.get("val/loss", float("nan")))
        self.logger.log(results, step)

        val_loss = results.get("val/loss", float("inf"))
        is_best = self.config.checkpoint.save_best and val_loss < self.state.best_val_loss

        # The aggregate can improve while one language gets worse, and then that
        # checkpoint is stored as "best". That is exactly what happened at the
        # end of the 1.3B runs: the mixture ran out of English, the model spent
        # its last steps on Korean only, and the Korean gain outvoted the English
        # regression. Selecting on the aggregate is still defensible, but the
        # trade has to be visible rather than hidden inside a mean.
        per_lang_now = {
            k.rsplit("_", 1)[-1]: v for k, v in results.items() if k.startswith("val/loss_")
        }
        regressed = sorted(
            lang for lang, v in per_lang_now.items()
            if v > self._best_per_lang.get(lang, float("inf"))
        )
        if is_best:
            self.state.best_val_loss = val_loss
            self._save(step, tag="best_val")
        for lang, v in per_lang_now.items():
            self._best_per_lang[lang] = min(self._best_per_lang.get(lang, float("inf")), v)

        # Surface validation on the console (a key milestone) even when metrics
        # otherwise go to W&B. Main process only, so multi-GPU doesn't duplicate.
        if is_main_process():
            per_lang = " ".join(f"{lang} {v:.3f}" for lang, v in per_lang_now.items())
            _console(
                f"[eval @ step {step}] val/loss {val_loss:.4f} "
                f"(ppl {results['val/perplexity']:.1f})"
                + (f" | {per_lang}" if per_lang else "")
                + ("  ← best" if is_best else "")
                + (
                    f"  ⚠ {'/'.join(regressed)} above its own best"
                    if regressed and is_best else ""
                )
            )
        self.model.train()

    @torch.no_grad()
    def _evaluate_downstream(self, step: int) -> None:
        """Score the frozen benchmark sets (spec 21.3).

        Deliberately does not touch ``best_val_loss``: 500 questions carry a
        standard error near 2%, so picking checkpoints on this would chase
        noise. Validation loss stays the selection criterion.
        """

        ev = self.config.evaluation
        if not (ev.downstream_enabled and self.downstream_tasks and self.tokenizer):
            return
        if step == getattr(self, "_last_downstream_step", None):
            return
        self._last_downstream_step = step

        from kkoma.evaluation.downstream import evaluate_task

        self.model.eval()
        results: dict = {}
        by_lang: dict[str, list[float]] = {}
        for name, language, examples in self.downstream_tasks:
            # Deal the questions round-robin across ranks. evaluate_task calls
            # the reduction exactly once per rank, including on a rank that
            # draws nothing, so the collective stays matched.
            shard = examples[self.dist.rank :: self.dist.world_size]
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                r = evaluate_task(
                    self.raw_model,
                    self.tokenizer,
                    shard,
                    self.device,
                    batch_size=ev.downstream_batch_size,
                    reduce_fn=all_reduce_sum,
                )
            results[f"downstream/{name}_acc_norm"] = r["acc_norm"]
            results[f"downstream/{name}_margin_max"] = r["margin_max"]
            results[f"downstream/{name}_margin_mean"] = r["margin_mean"]
            by_lang.setdefault(language, []).append(r["acc_norm"])

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        every = [a for accs in by_lang.values() for a in accs]
        if "en" in by_lang:
            results["downstream/en_avg"] = _mean(by_lang["en"])
        if "ko" in by_lang:
            results["downstream/ko_avg"] = _mean(by_lang["ko"])
        results["downstream/overall_avg"] = _mean(every)
        self.logger.log(results, step)

        if is_main_process():
            per_task = " ".join(
                f"{k.split('/')[1].rsplit('_acc_norm', 1)[0]} {v:.3f}"
                for k, v in results.items()
                if k.endswith("_acc_norm")
            )
            _console(
                f"[downstream @ step {step}] overall {results['downstream/overall_avg']:.3f}"
                + (f" | {per_task}" if per_task else "")
            )
        self.model.train()

    def _eval_batches_per_rank(self) -> int:
        """Per-rank batch cap so each validation stream consumes
        ``training.eval_tokens`` tokens in total, independent of world size
        (val losses stay comparable across GPU counts)."""

        per_batch = self.micro_bs * self.seq_len * self.dist.world_size
        return max(1, self.config.training.eval_tokens // per_batch)

    @torch.no_grad()
    def _eval_loss(self, loader: Iterable[dict], max_batches: Optional[int] = None) -> float:
        # Batch-weighted so a rank's empty shard contributes 0 without skipping
        # the collective. Under DDP each rank sees a disjoint shard; the SUM
        # reduce + division combines them into a single global mean.
        if max_batches is None:
            max_batches = self._eval_batches_per_rank()
        acc = torch.zeros(2, device=self.device)  # [sum of batch losses, count]
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            acc[0] += self._forward_loss(batch).detach()
            acc[1] += 1.0
        acc = all_reduce_sum(acc)  # every rank calls this exactly once
        if acc[1].item() == 0:
            return float("nan")
        return (acc[0] / acc[1]).item()

    def _save(self, step: int, tag: str) -> None:
        if not is_main_process():
            return
        path = os.path.join(self.config.checkpoint.dir, f"{tag}.pt")
        save_checkpoint(
            path,
            self.raw_model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            global_step=self.state.global_step,
            tokens_processed=self.state.tokens_processed,
            config_dict=self.config.to_dict(),
            extra={
                "best_val_loss": self.state.best_val_loss,
                "skipped_steps": self.state.skipped_steps,
                "wandb_run_id": self.logger.run_id,
                # Data sharding is world-size dependent, so a resume at a
                # different world size reads a different slice of the stream.
                # Recorded so scripts/_common.check_resume_compatibility can
                # refuse rather than silently diverge.
                "world_size": self.dist.world_size,
                # Streaming data position (spec 18.1/18.3): tokens_processed is
                # the position proxy; the flag records resume divergence.
                "data_position": {
                    "tokens_processed": self.state.tokens_processed,
                    "diverged_on_resume": self.state.data_position_diverged,
                },
                **self.provenance,
            },
        )
        if tag.startswith("step_"):
            rotate_checkpoints(self.config.checkpoint.dir, self.config.checkpoint.keep_last)

    def load_state(self, ckpt: dict) -> None:
        """Restore trainer counters from a loaded checkpoint dict."""

        self.state.global_step = ckpt.get("global_step", 0)
        self.state.tokens_processed = ckpt.get("tokens_processed", 0)
        self.state.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.state.skipped_steps = ckpt.get("skipped_steps", 0)
        # Safe default; run_training clears it when the data stream was
        # fast-forwarded to the exact consumed position (spec 18.3).
        self.state.data_position_diverged = True


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


__all__ = ["Trainer", "TrainState", "Logger"]
