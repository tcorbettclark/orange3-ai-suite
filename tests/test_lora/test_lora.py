"""orangecontrib.ai_suite.kernel.lora tests.

Data/artifact/metric-parsing tests always run; end-to-end training tests
require the locally converted model (demo/fetch_britllm.py) — skipped
elsewhere. They must run in plain Python without Orange/Qt (kernel rule).
"""

import json
import subprocess  # noqa: F401
from pathlib import Path

import pytest

from orangecontrib.ai_suite.kernel.lora import (
    LoRAAdapter,
    LoraData,
    LoRAModel,
    LoraOptions,
    stream_metrics,
    train_lora,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "britllm-4bit"
needs_britllm = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="converted model absent — run demo/fetch_britllm.py",
)
slow = pytest.mark.slow  # end-to-end runs (model load + iterations); excluded
# from the default suite via addopts -m "not slow"; run with `-m slow` when
# the converted model is present to validate the full path in one go.

SAMPLE_LOG_LINES = [
    "\x1b[1miter   train_loss     tok/s     tokens\x1b[0m",
    "\x1b[1;35m   1\x1b[0m    \x1b[1;35mval\x1b[0m \x1b[1m3.419\x1b[0m    \x1b[38;5;244m2.38s\x1b[0m",
    "\x1b[38;5;244m  20\x1b[0m    \x1b[1;32m0.725 \u25bc\x1b[0m    \x1b[1m  184\x1b[0m    \x1b[38;5;244m  3.3k\x1b[0m",
    "\x1b[38;5;244m  50\x1b[0m    \x1b[1;35mval\x1b[0m \x1b[1m0.018\x1b[0m    \x1b[38;5;244m1.40s\x1b[0m",
    "Model loaded: britllm/britllm-3b-v0.1 (some header noise line)",
    "Trainable parameters: 0.233% (6.21M / 2.67B)",
]


# ---- data staging (the two teaching formats) ------------------------------


def test_text_corpus_staging(tmp_path):
    data = LoraData.from_texts([f"document {i} text" for i in range(20)])
    assert data.kind == "text"
    staged = data.stage(tmp_path / "data")
    train_rows = [json.loads(line) for line in (staged / "train.jsonl").read_text().splitlines()]
    valid_rows = [json.loads(line) for line in (staged / "valid.jsonl").read_text().splitlines()]
    assert len(train_rows) + len(valid_rows) == 20
    assert all(set(r) == {"text"} for r in train_rows + valid_rows)
    assert len(train_rows) == 18 and len(valid_rows) == 2  # 10% -> val


def test_pair_corpus_staging_is_sft_with_masked_prompts(tmp_path):
    pairs = [(f"question {i}?", f"answer {i}.") for i in range(10)]
    data = LoraData.from_pairs(pairs)
    assert data.kind == "pairs"
    staged = data.stage(tmp_path / "data")
    rows = [json.loads(line) for line in (staged / "train.jsonl").read_text().splitlines()]
    assert all(set(r) == {"prompt", "completion"} for r in rows)
    # the SFT lesson: prompt and completion are separately authored
    assert rows[0] == {"prompt": pairs[0][0], "completion": pairs[0][1]}


def test_lora_data_guards():
    with pytest.raises(ValueError, match="at least 2 texts"):
        LoraData.from_texts(["one line only"])
    with pytest.raises(ValueError, match="at least 4"):
        LoraData.from_pairs([("q", "a"), ("q2", "a2"), ("q3", "a3")])


# ---- metric streaming (the widget's live-loss source) ----------------------


def test_stream_metrics_parses_the_real_log():
    seen = [stream_metrics(line) for line in SAMPLE_LOG_LINES]
    train = next(m for m in seen if m and m["kind"] == "train")
    val = next(m for m in seen if m and m["kind"] == "val")
    assert train == {"iter": 20, "loss": 0.725, "tok_s": 184, "kind": "train"}
    assert val == {"iter": 1, "loss": 3.419, "kind": "val"}
    params = next(m for m in seen if m and m["kind"] == "params")
    assert params == {"kind": "params", "pct": 0.233, "detail": "6.21M / 2.67B"}
    assert stream_metrics("Model loaded: header noise") is None


# ---- artifact round trip (adapter serialized with base model) --------------


def test_adapter_artifact_round_trip(tmp_path):
    adapter = LoRAAdapter(
        base_model="some/base-model",
        adapter_path=None,  # no trained weights yet: the raw base model
        hyperparameters={"iters": 300},
    )
    saved = adapter.save(tmp_path / "adapter")
    assert (saved / "lora_meta.json").exists()
    loaded = LoRAAdapter.load(saved)
    assert loaded.base_model == "some/base-model"
    assert loaded.hyperparameters["iters"] == 300


def test_adapter_without_adapters_loads_fine(tmp_path):
    # a copy of only the meta (adapter weights omitted) still loads as the
    # base model with adapter off: the raw-model comparison
    (tmp_path / "a").mkdir()
    LoRAAdapter(base_model="b/x", adapter_path=None, hyperparameters={}).save(tmp_path / "a")
    loaded = LoRAAdapter.load(tmp_path / "a")
    assert loaded.adapter_path is None


def test_model_import_requires_apple(monkeypatch, tmp_path):
    """Non-Apple machine: clear message, never a stack trace."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("mlx"):
            raise ImportError("No module named 'mlx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Apple Silicon"):
        LoRAModel("some/model")


# ---- end-to-end (converted model required) ---------------------------------


@needs_britllm
@slow
def test_train_lora_tiny_run(tmp_path):
    """2 iterations on the converted 4-bit model: adapter produced, model
    generates, save/load round trip reproduces identical greedy output."""
    from orangecontrib.ai_suite.kernel.lora import train_lora

    data = LoraData.from_texts([f"the quick brown fox jumped over fence {i}" for i in range(8)])
    adapter = train_lora(
        MODEL_DIR,
        data,
        tmp_path / "adapter",
        options=LoraOptions(iters=2, batch_size=1, max_seq_length=128, num_layers=4),
    )
    assert (Path(adapter.adapter_path) / "adapters.safetensors").exists()
    assert adapter.hyperparameters["iters"] == 2

    model = LoRAModel(adapter.base_model, adapter.adapter_path)
    prompt = "the quick brown fox"
    out_on = model.generate(prompt, temperature=0, max_new_tokens=8, seed=0)
    assert isinstance(out_on, str) and out_on.startswith(prompt)

    # adapter OFF = the raw base model, deterministic
    model.adapter_enabled = False
    out_off = model.generate(prompt, temperature=0, max_new_tokens=8, seed=0)
    assert isinstance(out_off, str)
    # determinism
    assert model.generate(prompt, temperature=0, max_new_tokens=8, seed=0) == out_off

    # artifact round trip: greedy generations identical to the trained setup
    saved = model.save(tmp_path / "artifact")
    reloaded = LoRAModel.from_artifact(saved)
    assert reloaded.adapter_enabled  # artifact carries the adapter
    assert reloaded.generate(prompt, temperature=0, max_new_tokens=8, seed=0) == out_on


@needs_britllm
@slow
def test_train_lora_greedy_determinism(tmp_path):
    data = LoraData.from_texts(["alpha beta gamma delta." for _ in range(6)])
    a = train_lora(
        MODEL_DIR,
        data,
        tmp_path / "a1",
        options=LoraOptions(iters=2, batch_size=1, max_seq_length=64, num_layers=2, seed=7),
    )
    m1 = LoRAModel(a.base_model, a.adapter_path)
    prompt, kwargs = "alpha", {"temperature": 1.0, "max_new_tokens": 10, "seed": 3}
    first = m1.generate(prompt, **kwargs)
    second = m1.generate(prompt, **kwargs)
    assert first == second  # seeded sampling is deterministic


def test_valid_fraction_guards():
    texts = [f"document {i}" for i in range(5)]
    pairs = [(f"question {i}?", f"answer {i}.") for i in range(5)]
    with pytest.raises(ValueError, match="valid_fraction"):
        LoraData.from_texts(texts, valid_fraction=1.0)
    with pytest.raises(ValueError, match="valid_fraction"):
        LoraData.from_pairs(pairs, valid_fraction=-0.1)
    # even an extreme fraction leaves at least one training row
    data = LoraData.from_texts(texts, valid_fraction=0.99)
    assert len(data.train) == 1 and len(data.valid) == 4
    # zero means no validation split
    assert LoraData.from_texts(texts, valid_fraction=0.0).valid == []
