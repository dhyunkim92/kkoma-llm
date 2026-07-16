"""Shared wiring for the training scripts.

Builds tokenizer, data loaders, model, optimizer, scheduler, and the trainer
from a ``RunConfig``, so ``train_architecture``, ``train_pretraining``, and
``train_continued`` only differ by their YAML config and a couple of flags.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from kkoma.config import (
    DataSource,
    RunConfig,
    resolve_grad_accum,
    tokens_per_optimizer_step,
)
from kkoma.data.mixture import MixtureStream
from kkoma.data.packing import PackedDataset, collate_blocks
from kkoma.data.preprocessing import CleanConfig
from kkoma.model.model import KkomaModel
from kkoma.tokenizer.utils import KkomaTokenizer
from kkoma.training.checkpoint import (
    apply_checkpoint,
    collect_provenance,
    load_checkpoint,
    read_checkpoint,
)
from kkoma.training.distributed import (
    DistInfo,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from kkoma.training.optimizer import build_optimizer
from kkoma.training.scheduler import build_scheduler
from kkoma.training.trainer import Logger, Trainer


def set_seeds(config: RunConfig) -> None:
    import random

    seed = config.project.seed
    random.seed(seed)
    torch.manual_seed(config.project.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.project.init_seed)
    try:
        import numpy as np

        np.random.seed(config.project.data_seed)
    except Exception:
        pass


def pick_device(dist: DistInfo) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist.local_rank}")
    return torch.device("cpu")


def build_tokenizer(config: RunConfig) -> KkomaTokenizer:
    tokenizer = KkomaTokenizer.from_file(config.tokenizer.path)
    if len(tokenizer) != config.model.vocab_size:
        raise ValueError(
            f"tokenizer at {config.tokenizer.path} has vocab {len(tokenizer)} "
            f"but model.vocab_size is {config.model.vocab_size}"
        )
    return tokenizer


def _lang_tag(source: DataSource) -> str:
    blob = f"{source.name} {source.subset or ''}".lower()
    return "ko" if ("ko" in blob or "korean" in blob) else "en"


def build_train_loader(
    config: RunConfig, tokenizer: KkomaTokenizer, dist: DistInfo, skip_blocks: int = 0
) -> DataLoader:
    clean = CleanConfig(min_chars=config.data.min_doc_chars)
    stream = MixtureStream(config.data.sources, seed=config.data.sampling_seed, clean_config=clean)
    dataset = PackedDataset(
        documents=stream,
        tokenizer=tokenizer,
        block_size=config.model.context_length,
        rank=dist.rank,
        world_size=dist.world_size,
        skip_blocks=skip_blocks,
    )
    return DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        collate_fn=collate_blocks,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_val_loaders(
    config: RunConfig, tokenizer: KkomaTokenizer, dist: Optional[DistInfo] = None
) -> dict:
    """Group validation sources by language tag (en/ko).

    Under DDP each rank gets a disjoint shard of the validation stream so eval
    isn't computed redundantly on every GPU; the Trainer's SUM-reduce combines
    the per-rank partials into one global figure.
    """

    if not config.data.val_sources:
        return {}
    rank = dist.rank if dist else 0
    world_size = dist.world_size if dist else 1

    by_lang: dict[str, list[DataSource]] = {}
    for src in config.data.val_sources:
        by_lang.setdefault(_lang_tag(src), []).append(src)

    clean = CleanConfig(min_chars=config.data.min_doc_chars)
    loaders = {}
    for lang, sources in by_lang.items():
        stream = MixtureStream(sources, seed=config.data.sampling_seed, clean_config=clean)
        dataset = PackedDataset(
            stream, tokenizer, config.model.context_length,
            rank=rank, world_size=world_size,
        )
        loaders[lang] = DataLoader(
            dataset,
            batch_size=config.training.micro_batch_size,
            collate_fn=collate_blocks,
            num_workers=0,
        )
    return loaders


def build_downstream_tasks(config: RunConfig) -> list:
    """Load the frozen benchmark sets named in the config (spec 21.3).

    Returns ``[(name, language, examples), ...]``. A task whose file is missing
    is reported and skipped rather than raising: the sets come from a separate
    prepare step, and a run should not die at step 0 because someone has not
    fetched them yet.
    """

    from kkoma.evaluation.downstream import load_examples

    ev = config.evaluation
    if not ev.downstream_enabled:
        return []

    tasks = []
    for task in ev.downstream_tasks:
        if not task.enabled:
            continue
        if not os.path.exists(task.path):
            if is_main_process():
                print(
                    f"[downstream] {task.name}: {task.path} not found; skipping "
                    "(run scripts/prepare_downstream_data.py)"
                )
            continue
        tasks.append((task.name, task.language, load_examples(task.path)))
    return tasks


def build_model(config: RunConfig, device: torch.device) -> KkomaModel:
    model = KkomaModel(config.model).to(device)
    return model


def maybe_wrap_ddp(model: KkomaModel, dist: DistInfo, config: RunConfig):
    if not dist.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [dist.local_rank] if torch.cuda.is_available() else None
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=config.distributed.find_unused_parameters,
    )


def run_training(
    config: RunConfig,
    resume_path: Optional[str] = None,
    init_from: Optional[str] = None,
) -> None:
    """End-to-end training entrypoint shared by all train scripts.

    ``init_from`` loads only model weights (used for continued pretraining);
    ``resume_path`` restores the full training state.
    """

    dist = setup_distributed(config.distributed.backend)
    set_seeds(config)
    device = pick_device(dist)

    if is_main_process():
        os.makedirs(config.checkpoint.dir, exist_ok=True)
        config.save(os.path.join(config.checkpoint.dir, "run_config.json"))

    tokenizer = build_tokenizer(config)
    raw_model = build_model(config, device)

    # Continued pretraining: load Base weights, fresh optimizer state (v1
    # default). restore_rng=False keeps the seeds set_seeds just applied
    # instead of adopting the base run's end-of-training RNG.
    if init_from:
        load_checkpoint(
            init_from, raw_model, optimizer=None, scheduler=None, scaler=None,
            map_location=str(device), restore_rng=False,
        )
        if is_main_process():
            print(f"[init] loaded base weights from {init_from}")

    optimizer = build_optimizer(raw_model, config.optimizer)

    # Scheduler horizon must match the optimizer steps the Trainer actually runs;
    # both use the same resolver so the LR schedule spans the real run.
    tokens_per_step = tokens_per_optimizer_step(config, dist.world_size)
    total_steps = max(1, config.training.max_tokens // max(tokens_per_step, 1))
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps)

    model = maybe_wrap_ddp(raw_model, dist, config)

    # Read the resume payload before the Logger and data loaders exist so the
    # checkpointed W&B run id can be reused (spec section 23) and the data
    # stream can be fast-forwarded to the consumed position (spec 18.3).
    ckpt = None
    skip_blocks = 0
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"--resume checkpoint not found: {resume_path} "
                "(refusing to silently restart from step 0)"
            )
        ckpt = read_checkpoint(resume_path, map_location=str(device))
        if ckpt.get("wandb_run_id") and not config.logging.wandb_run_id:
            config.logging.wandb_run_id = ckpt["wandb_run_id"]
        if config.training.resume_fastforward:
            # Every optimizer step (applied or scaler-skipped) consumed exactly
            # grad_accum * micro_batch_size blocks per rank; the stream is
            # deterministic, so skipping that prefix lands on the exact next batch.
            steps_consumed = ckpt.get("global_step", 0) + ckpt.get("skipped_steps", 0)
            skip_blocks = (
                steps_consumed
                * resolve_grad_accum(config, dist.world_size)
                * config.training.micro_batch_size
            )

    train_loader = build_train_loader(config, tokenizer, dist, skip_blocks=skip_blocks)
    val_loaders = build_val_loaders(config, tokenizer, dist)
    logger = Logger(config, dist)

    trainer = Trainer(
        config=config,
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        dist=dist,
        device=device,
        val_loaders=val_loaders,
        logger=logger,
        provenance=collect_provenance(config),
        tokenizer=tokenizer,
        downstream_tasks=build_downstream_tasks(config),
    )

    if ckpt is not None:
        apply_checkpoint(ckpt, raw_model, optimizer, scheduler, trainer.scaler)
        trainer.load_state(ckpt)
        trainer.state.data_position_diverged = skip_blocks == 0
        if is_main_process():
            print(f"[resume] continuing from step {trainer.state.global_step}")
            if skip_blocks:
                print(
                    f"[resume] fast-forwarding the data stream by {skip_blocks} "
                    "blocks per rank (re-tokenizes the consumed prefix once; "
                    "disable with training.resume_fastforward: false)"
                )
            else:
                print(
                    "[resume] data stream restarts from the beginning; exact "
                    "position not reproduced (recorded in checkpoints, spec 18.3)"
                )

    try:
        trainer.train()
    except BaseException:
        # Skip the barrier: peers may be stuck inside a collective this rank
        # never joined, and waiting on them would hide the real error.
        cleanup_distributed(barrier=False)
        raise
    else:
        cleanup_distributed()


__all__ = ["run_training", "build_tokenizer", "build_model", "set_seeds", "pick_device"]
