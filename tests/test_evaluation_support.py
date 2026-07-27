"""Support modules behind the reported numbers: efficiency, init, metrics, cleaning.

None of these had tests before the 2026-07 audit, and each produces or shapes a
number that ends up in an artifact: `efficiency.py` fills the `efficiency`
section of every evaluation JSON, `initialization.py` decides the scale every
run starts from, `metrics.py` feeds `train/tokens_per_second` and the memory
gauges, and `preprocessing.py` decides which documents enter the corpus at all.
"""

from __future__ import annotations

import math

import pytest
import torch

from kkoma.data.preprocessing import CleanConfig, clean_document, strip_control_chars
from kkoma.evaluation.efficiency import attention_kv_cache_bytes, benchmark_forward, estimate_flops_per_token
from kkoma.model.initialization import apply_init
from kkoma.model.model import KkomaModel
from kkoma.training.metrics import Throughput, grad_global_norm, perplexity
from tests.conftest import tiny_config


# ---------------------------------------------------------------------------
# efficiency.py
# ---------------------------------------------------------------------------


def test_flops_per_token_is_six_times_non_embedding_params():
    """The 6N approximation. It is only meaningful if N excludes embeddings —
    counting them would inflate a small model's FLOPs by a third."""

    model = KkomaModel(tiny_config())
    assert estimate_flops_per_token(model) == pytest.approx(
        6.0 * model.num_parameters(non_embedding=True)
    )


def test_kv_cache_bytes_scales_with_kv_heads_not_query_heads():
    """This is the whole point of GQA: the cache shrinks with n_kv_head."""

    mha = tiny_config()
    mha.n_kv_head = mha.n_head
    gqa = tiny_config()
    gqa.n_kv_head = mha.n_head // 2

    big = attention_kv_cache_bytes(mha, batch=1, seq_len=128)
    small = attention_kv_cache_bytes(gqa, batch=1, seq_len=128)
    assert small == pytest.approx(big / 2)


def test_benchmark_forward_reports_the_window_it_measured():
    """`peak_allocated_mb` covers only this call — it resets the peak counter
    first, so it is not the peak of whatever ran before it. The audit found that
    number being read as a whole-evaluation peak."""

    model = KkomaModel(tiny_config()).eval()
    r = benchmark_forward(model, torch.device("cpu"), batch_size=2, seq_len=16, steps=2, warmup=1)
    assert r["tokens_per_second"] > 0
    assert r["step_time_ms"] > 0
    assert r["parameters"] == model.num_parameters()
    assert r["non_embedding_parameters"] < r["parameters"]
    assert r["peak_allocated_mb"] == 0.0  # CPU has no CUDA allocator to read


# ---------------------------------------------------------------------------
# initialization.py
# ---------------------------------------------------------------------------


def test_residual_projections_are_scaled_down_by_depth():
    """o_proj and the FFN output are redrawn at std/sqrt(2*n_layer) so the
    residual stream does not grow with depth. The tag that selects them is an
    attribute set in a factory, so it can be dropped silently — this is what
    would catch that."""

    cfg = tiny_config()
    cfg.n_layer = 8
    torch.manual_seed(0)
    model = KkomaModel(cfg)

    expected = cfg.initializer_range / math.sqrt(2 * cfg.n_layer)
    residual = [p for n, p in model.named_parameters()
                if n.endswith("o_proj.weight") or n.endswith("down_proj.weight")]
    assert residual, "no residual projections found — the tag mechanism changed"
    for p in residual:
        assert p.std().item() == pytest.approx(expected, rel=0.25)

    # A non-residual projection keeps the plain std, so the two are distinguishable.
    q = dict(model.named_parameters())["blocks.0.attn.q_proj.weight"]
    assert q.std().item() == pytest.approx(cfg.initializer_range, rel=0.25)


def test_norm_weights_are_left_at_one():
    model = KkomaModel(tiny_config())
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1:
            assert torch.allclose(p, torch.ones_like(p)), name


def test_apply_init_is_deterministic_given_a_seed():
    """Two apply_init passes under the same seed must agree.

    Note the seed has to be set immediately before apply_init, not before
    construction: building the modules draws from the same generator (torch's
    default nn.Linear init) before apply_init overwrites the result, so
    `manual_seed(s); KkomaModel(cfg)` and `manual_seed(s); apply_init(model)`
    start from different positions in the stream.
    """

    cfg = tiny_config()
    model = KkomaModel(cfg)

    torch.manual_seed(3)
    apply_init(model, cfg)
    first = {n: p.detach().clone() for n, p in model.named_parameters()}

    torch.manual_seed(3)
    apply_init(model, cfg)
    for n, p in model.named_parameters():
        assert torch.equal(p, first[n]), n


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_throughput_rate_is_tokens_over_elapsed():
    t = Throughput()
    t.update(1000)
    assert t.rate() > 0


def test_grad_global_norm_matches_manual_l2():
    model = KkomaModel(tiny_config())
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    _, loss = model(ids, labels=ids)
    loss.backward()

    manual = math.sqrt(sum(
        (p.grad.float() ** 2).sum().item() for p in model.parameters() if p.grad is not None
    ))
    assert grad_global_norm(model.parameters()) == pytest.approx(manual, rel=1e-5)


def test_perplexity_saturates_instead_of_overflowing():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(2.0) == pytest.approx(math.exp(2.0))
    assert math.isinf(perplexity(1e4))  # exp() would raise OverflowError


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------


def test_clean_document_drops_short_and_empty():
    cfg = CleanConfig(min_chars=16)
    assert clean_document("", cfg) is None
    assert clean_document("   ", cfg) is None
    assert clean_document("too short", cfg) is None
    long_enough = "this document is definitely long enough to survive"
    assert clean_document(long_enough, cfg) == long_enough


def test_clean_document_strips_control_characters():
    cfg = CleanConfig(min_chars=4)
    out = clean_document("hello\x00\x07 world and some more text", cfg)
    assert out is not None and "\x00" not in out and "\x07" not in out


def test_strip_control_chars_keeps_newline_and_tab():
    assert strip_control_chars("a\nb\tc") == "a\nb\tc"


def test_min_chars_counts_characters_not_bytes():
    """16 Korean characters are ~48 UTF-8 bytes; the threshold must not depend
    on encoding or the two corpora get filtered at different content levels."""

    cfg = CleanConfig(min_chars=16)
    korean = "가나다라마바사아자차카타파하거너"  # exactly 16 characters
    assert len(korean) == 16
    assert clean_document(korean, cfg) == korean


# ---------------------------------------------------------------------------
# evaluation/generation.py and the HF streaming branch of data/streaming.py
# ---------------------------------------------------------------------------


def test_generate_samples_returns_one_record_per_prompt(tiny_tokenizer):
    """The `generation` section of every evaluation JSON comes from here."""

    from kkoma.evaluation.generation import generate_samples

    cfg = tiny_config(vocab_size=len(tiny_tokenizer), context_length=64)
    model = KkomaModel(cfg).eval()
    prompts = ["The model", "이 모델은"]

    out = generate_samples(model, tiny_tokenizer, torch.device("cpu"),
                           prompts=prompts, max_new_tokens=4, seed=7)
    assert [s["prompt"] for s in out] == prompts
    for s in out:
        assert isinstance(s["completion"], str)
        assert len(s["token_ids"]) > 0


def test_generate_samples_is_reproducible_under_a_seed(tiny_tokenizer):
    from kkoma.evaluation.generation import generate_samples

    cfg = tiny_config(vocab_size=len(tiny_tokenizer), context_length=64)
    model = KkomaModel(cfg).eval()
    kw = dict(prompts=["The model"], max_new_tokens=6, seed=11)
    a = generate_samples(model, tiny_tokenizer, torch.device("cpu"), **kw)
    b = generate_samples(model, tiny_tokenizer, torch.device("cpu"), **kw)
    assert a[0]["token_ids"] == b[0]["token_ids"]


def test_hf_source_passes_revision_subset_and_split_through(monkeypatch):
    """The HF branch needs a network, so verify the call we make, not the data.

    `revision` is what makes a streamed corpus reproducible; the audit noted it
    is never set by any prepare script, and nothing checked it was even wired
    through to `load_dataset`.
    """

    import datasets

    from kkoma.config import DataSource
    from kkoma.data.streaming import stream_source

    captured = {}

    class FakeDS:
        def __init__(self, rows):
            self.rows = rows
            self.shuffled = None

        def shuffle(self, seed, buffer_size):
            self.shuffled = (seed, buffer_size)
            return self

        def __iter__(self):
            return iter(self.rows)

    def fake_load_dataset(name, subset, split=None, revision=None, streaming=None):
        captured.update(name=name, subset=subset, split=split,
                        revision=revision, streaming=streaming)
        return FakeDS([{"text": "a streamed document long enough to survive cleaning"}])

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    src = DataSource(name="HuggingFaceFW/fineweb-2", subset="kor_Hang",
                     split="train", revision="deadbeef", text_key="text")
    docs = list(stream_source(src))

    assert captured == {
        "name": "HuggingFaceFW/fineweb-2", "subset": "kor_Hang",
        "split": "train", "revision": "deadbeef", "streaming": True,
    }
    assert docs == ["a streamed document long enough to survive cleaning"]


def test_hf_source_shuffles_with_the_configured_buffer(monkeypatch):
    import datasets

    from kkoma.config import DataSource
    from kkoma.data.streaming import stream_source

    seen = {}

    class FakeDS:
        def shuffle(self, seed, buffer_size):
            seen.update(seed=seed, buffer_size=buffer_size)
            return self

        def __iter__(self):
            return iter([{"text": "another sufficiently long streamed document"}])

    monkeypatch.setattr(datasets, "load_dataset",
                        lambda *a, **k: FakeDS())

    src = DataSource(name="repo", text_key="text")
    list(stream_source(src, shuffle_seed=5, shuffle_buffer=1000))
    assert seen == {"seed": 5, "buffer_size": 1000}
