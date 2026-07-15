"""Weight tying tests (spec 11): shared Parameter object, persists across I/O."""

from __future__ import annotations

import torch

from kkoma.model.model import KkomaModel
from kkoma.training.checkpoint import load_checkpoint, save_checkpoint
from tests.conftest import tiny_config


def test_lm_head_is_embedding(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.token_embedding.weight


def test_tying_in_optimizer_param_list(tiny_model):
    from kkoma.config import OptimizerConfig
    from kkoma.training.optimizer import build_optimizer

    opt = build_optimizer(tiny_model, OptimizerConfig())
    params = [p for group in opt.param_groups for p in group["params"]]
    # The shared tensor must appear exactly once.
    shared = tiny_model.token_embedding.weight
    assert sum(1 for p in params if p is shared) == 1


def test_tying_survives_save_load(tmp_path):
    torch.manual_seed(0)
    model = KkomaModel(tiny_config())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        path, model, opt, None, None,
        global_step=0, tokens_processed=0, config_dict={},
    )

    torch.manual_seed(1)
    model2 = KkomaModel(tiny_config())
    load_checkpoint(path, model2)
    assert model2.lm_head.weight is model2.token_embedding.weight
    assert torch.equal(model2.lm_head.weight, model.lm_head.weight)


def test_no_separate_lm_head_when_untied():
    cfg = tiny_config(tie_word_embeddings=False)
    model = KkomaModel(cfg)
    assert model.lm_head.weight is not model.token_embedding.weight
