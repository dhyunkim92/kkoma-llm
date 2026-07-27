"""Checkpoint save / resume (spec section 18).

A checkpoint stores everything needed to resume bit-for-bit (modulo streaming
data position): model, optimizer, scheduler, grad scaler, step counters, RNG
states, config, and provenance identifiers. Weight tying is re-established
after load so the LM head and token embedding remain the same Parameter object.
"""

from __future__ import annotations

import glob
import hashlib
import os
import subprocess
from typing import Optional

import torch

from kkoma.training.distributed import is_main_process


def _rng_states() -> dict:
    states = {
        "python": None,  # filled by trainer if desired
        "torch": torch.get_rng_state(),
        "numpy": None,
    }
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng(states: dict) -> None:
    """Restore generator states saved by ``_rng_states``.

    Every state has to be a CPU ByteTensor. ``read_checkpoint`` is called with
    ``map_location`` set to the training device, and that moves *every* tensor
    in the payload — the RNG states included — so on GPU they arrive as CUDA
    tensors and ``set_rng_state`` rejects them. Resuming on GPU used to die here
    with "RNG state must be a torch.ByteTensor", which named the symptom and not
    the cause; no shipped run ever hit it because none used --resume.
    """

    def _cpu_byte(t):
        return t.detach().to("cpu", torch.uint8) if torch.is_tensor(t) else t

    if states.get("torch") is not None:
        torch.set_rng_state(_cpu_byte(states["torch"]))
    if states.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(states["numpy"])
        except Exception:
            pass
    if states.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [_cpu_byte(s) for s in states["cuda"]]
        # A checkpoint from a different GPU count cannot be applied per-device.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    *,
    global_step: int,
    tokens_processed: int,
    config_dict: dict,
    extra: Optional[dict] = None,
) -> None:
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "tokens_processed": tokens_processed,
        "rng": _rng_states(),
        "config": config_dict,
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """Load the raw checkpoint payload without applying any of it.

    Lets callers inspect metadata (wandb run id, counters) before the training
    objects that ``apply_checkpoint`` needs even exist.
    """

    return torch.load(path, map_location=map_location, weights_only=False)


def apply_checkpoint(
    ckpt: dict,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
) -> dict:
    target = _unwrap(model)
    target.load_state_dict(ckpt["model"])
    # Re-tie embeddings to keep a single shared Parameter object after load.
    if getattr(target.config, "tie_word_embeddings", False):
        target.lm_head.weight = target.token_embedding.weight

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        _restore_rng(ckpt["rng"])
    return ckpt


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    ckpt = read_checkpoint(path, map_location=map_location)
    return apply_checkpoint(
        ckpt, model, optimizer, scheduler, scaler, restore_rng=restore_rng
    )


def _file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT", "")


def collect_provenance(config) -> dict:
    """Provenance identifiers stored in every checkpoint (spec sections 18.1, 25).

    ``tokenizer_id`` is the sha256 of ``tokenizer.json``; ``data_manifests``
    maps each source directory's ``data_manifest.json`` to its sha256, tying
    the run to the exact prepared corpus.
    """

    tok_json = os.path.join(config.tokenizer.path, "tokenizer.json")
    manifests = {}
    sources = list(config.data.sources) + list(config.data.val_sources or [])
    for src in sources:
        if not src.path:
            continue
        manifest = os.path.join(os.path.dirname(src.path), "data_manifest.json")
        if manifest not in manifests:
            digest = _file_sha256(manifest)
            if digest is not None:
                manifests[manifest] = digest

    env = {"torch": torch.__version__}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return {
        "git_commit": _git_commit(),
        "tokenizer_id": {"path": config.tokenizer.path, "sha256": _file_sha256(tok_json)},
        "data_manifests": manifests,
        "env": env,
    }


def rotate_checkpoints(checkpoint_dir: str, keep_last: int, pattern: str = "step_*.pt") -> None:
    """Keep only the most recent ``keep_last`` periodic checkpoints."""

    if not is_main_process() or keep_last <= 0:
        return
    files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    for stale in files[:-keep_last]:
        try:
            os.remove(stale)
        except OSError:
            pass


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "read_checkpoint",
    "apply_checkpoint",
    "collect_provenance",
    "rotate_checkpoints",
]
