"""Every shipped YAML must load strictly; typos and orphan configs fail here."""

from __future__ import annotations

import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkoma.config import RunConfig
from kkoma.tokenizer.train import TokenizerTrainConfig

pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CONFIGS = sorted(
    p for p in glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)
    if os.sep + "tokenizer" + os.sep not in p
)
TOKENIZER_CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "tokenizer", "*.yaml")))


def test_config_inventory_present():
    assert len(RUN_CONFIGS) >= 13  # 10 architecture + 3 base + continued
    assert TOKENIZER_CONFIGS


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_run_config_loads_strictly(path):
    config = RunConfig.from_yaml(path)  # raises on any unknown key
    stem = os.path.splitext(os.path.basename(path))[0]
    # run_name and checkpoint dir must identify the file they came from.
    assert stem.replace("_", "-") in config.project.run_name.replace("_", "-")
    assert config.model.vocab_size == 32768
    assert config.data.sources, f"{stem} has no train sources"
    assert config.data.val_sources, f"{stem} has no validation sources"


def test_run_names_and_checkpoint_dirs_unique():
    names = [RunConfig.from_yaml(p).project.run_name for p in RUN_CONFIGS]
    dirs = [RunConfig.from_yaml(p).checkpoint.dir for p in RUN_CONFIGS]
    assert len(set(names)) == len(names)
    assert len(set(dirs)) == len(dirs)


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("model:\n  n_kv_heads: 4\n", encoding="utf-8")  # typo: n_kv_head
    with pytest.raises(ValueError, match="n_kv_heads"):
        RunConfig.from_yaml(str(bad))


@pytest.mark.parametrize("path", TOKENIZER_CONFIGS, ids=os.path.basename)
def test_tokenizer_config_wired(path):
    cfg = TokenizerTrainConfig.from_yaml(path, input_files=["x.jsonl"])
    assert cfg.vocab_size == 32768
    assert cfg.output_dir

    from scripts.prepare_tokenizer_data import sources_from_config

    sources, total_gb = sources_from_config(path)
    assert len(sources) == 2 and total_gb > 0
    assert abs(sum(s.weight for s in sources) - 1.0) < 1e-6
