"""orangecontrib.ai_suite.kernel.lora tests.

Data/artifact/metric-parsing tests always run; end-to-end training tests
require the locally converted model (demo/fetch_britllm.py) — skipped
elsewhere. They must run in plain Python without Orange/Qt (kernel rule).
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from orangecontrib.ai_suite.kernel.assets.britllm import BRITLLM_4BIT
from orangecontrib.ai_suite.kernel.assets.hf import cached_path
from orangecontrib.ai_suite.kernel.lora import (
    LoRAAdapter,
    LoRAData,
    LoRAModel,
    LoRAOptions,
    parse_metric,
    split_for_training,
    train_lora,
)
from orangecontrib.ai_suite.kernel.lora.model import _echo, _segments
from orangecontrib.ai_suite.kernel.lora.training import _adapter_weights, _lora_command, _stage

_NO_MODEL = (
    "pinned 4-bit model not cached — use the BritLLM Models widget or examples/fetch_britllm.py"
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
    "Trainable parameters: 0.233% (6.210M/2667.000M)",
]


# ---- data staging (the two teaching formats) ------------------------------


def test_text_corpus_staging(tmp_path):
    data = LoRAData.from_texts([f"document {i} text" for i in range(20)])
    assert data.format == "text"
    assert len(data.rows) == 20
    train, valid = data.split(valid_fraction=0.1, seed=0)
    assert len(train) == 18 and len(valid) == 2  # 10% -> val
    staged = _stage(data.format, train, valid, tmp_path / "data")
    train_rows = [json.loads(line) for line in (staged / "train.jsonl").read_text().splitlines()]
    valid_rows = [json.loads(line) for line in (staged / "valid.jsonl").read_text().splitlines()]
    assert len(train_rows) + len(valid_rows) == 20
    assert all(set(r) == {"text"} for r in train_rows + valid_rows)


def test_pair_corpus_staging_is_sft(tmp_path):
    pairs = [(f"question {i}?", f"answer {i}.") for i in range(10)]
    data = LoRAData.from_pairs(pairs)
    assert data.format == "pairs"
    train, valid = data.split(valid_fraction=0.1, seed=0)
    staged = _stage(data.format, train, valid, tmp_path / "data")
    rows = [
        json.loads(line)
        for name in ("train", "valid")
        for line in (staged / f"{name}.jsonl").read_text().splitlines()
    ]
    assert all(set(r) == {"prompt", "completion"} for r in rows)
    # the SFT lesson: prompt and completion are separately authored, and the
    # staged rows are exactly the pairs that went in
    assert sorted((r["prompt"], r["completion"]) for r in rows) == sorted(pairs)


def test_split_depends_on_content_not_arrival_order():
    texts = [f"document {i}" for i in range(40)]
    first = LoRAData.from_texts(texts).split(valid_fraction=0.25, seed=3)
    shuffled = LoRAData.from_texts(list(reversed(texts))).split(valid_fraction=0.25, seed=3)
    assert set(first[0]) == set(shuffled[0])
    assert set(first[1]) == set(shuffled[1])


def test_lora_data_guards():
    with pytest.raises(ValueError, match="at least 2 texts"):
        LoRAData.from_texts(["one line only"])
    with pytest.raises(ValueError, match="at least 4"):
        LoRAData.from_pairs([("q", "a"), ("q2", "a2"), ("q3", "a3")])


def test_rows_must_match_the_declared_format():
    with pytest.raises(ValueError, match="unknown format"):
        LoRAData("json", ["x"])
    with pytest.raises(TypeError, match="text rows"):
        LoRAData("text", [("a", "b")])
    with pytest.raises(TypeError, match="pair rows"):
        LoRAData("pairs", ["x"])


def test_split_guards_and_edges():
    data = LoRAData.from_texts([f"document {i}" for i in range(5)])
    with pytest.raises(ValueError, match="valid_fraction"):
        data.split(valid_fraction=1.0, seed=0)
    with pytest.raises(ValueError, match="valid_fraction"):
        data.split(valid_fraction=-0.1, seed=0)
    # even an extreme fraction leaves at least one training row
    train, valid = data.split(valid_fraction=0.99, seed=0)
    assert len(train) == 1 and len(valid) == 4
    # zero means no validation split
    assert data.split(valid_fraction=0.0, seed=0)[1] == []


def test_split_for_training_refuses_a_small_validation_set():
    data = LoRAData.from_texts([f"document {i}" for i in range(20)])
    with pytest.raises(ValueError, match="valid split has 2 rows"):
        split_for_training(data, LoRAOptions(batch_size=4, validation_fraction=0.1))
    train, valid = split_for_training(data, LoRAOptions(batch_size=1, validation_fraction=0.1))
    assert len(train) == 18 and len(valid) == 2
    _, valid = split_for_training(data, LoRAOptions(validation_fraction=0.0))
    assert valid == []


def test_no_validation_split_omits_valid_file(tmp_path):
    # mlx-lm treats a missing valid.jsonl as no validation, but fails on an empty one
    staged = _stage("text", ["one", "two"], [], tmp_path / "data")
    assert (staged / "train.jsonl").exists()
    assert not (staged / "valid.jsonl").exists()
    # a stale file from an earlier run must not be picked up
    (staged / "valid.jsonl").write_text("stale\n")
    _stage("text", ["one", "two"], [], staged)
    assert not (staged / "valid.jsonl").exists()


def test_options_validate_and_are_frozen():
    for name, value in (
        ("iters", 0),
        ("batch_size", 0),
        ("num_layers", 0),
        ("save_every", 0),
    ):
        with pytest.raises(ValueError, match=name):
            LoRAOptions(**{name: value})
    with pytest.raises(ValueError, match="learning_rate"):
        LoRAOptions(learning_rate=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        LoRAOptions(validation_fraction=1.0)
    with pytest.raises(FrozenInstanceError):
        LoRAOptions().iters = 1


# ---- metric streaming (the widget's live-loss source) ----------------------


def test_parse_metric_parses_the_real_log():
    seen = [parse_metric(line) for line in SAMPLE_LOG_LINES]
    train = next(m for m in seen if m and m["kind"] == "train")
    val = next(m for m in seen if m and m["kind"] == "val")
    assert train == {"iter": 20, "loss": 0.725, "tok_s": 184, "kind": "train"}
    assert val == {"iter": 1, "loss": 3.419, "kind": "val"}
    params = next(m for m in seen if m and m["kind"] == "params")
    assert params == {"kind": "params", "pct": 0.233, "detail": "6.210M/2667.000M"}
    assert parse_metric("Model loaded: header noise") is None


def test_parse_metric_accepts_a_thousands_separator():
    # mlx-lm formats tokens/s with "{:,.0f}"
    assert parse_metric("  30    0.700 \u25bc   1,184    6.7k")["tok_s"] == 1184


# ---- the mlx-lm command -----------------------------------------------------


def test_lora_command_passes_the_options(tmp_path):
    options = LoRAOptions(
        iters=7,
        learning_rate=2e-5,
        num_layers=3,
        batch_size=2,
        max_seq_length=64,
        steps_per_report=1,
        steps_per_eval=2,
        save_every=3,
        seed=11,
    )
    cmd = _lora_command("base", tmp_path / "data", tmp_path / "adapter", options, "pairs")
    value = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    assert value("--model") == "base"
    assert value("--iters") == "7"
    assert value("--learning-rate") == "2e-05"
    assert value("--num-layers") == "3"
    assert value("--batch-size") == "2"
    assert value("--max-seq-length") == "64"
    assert value("--steps-per-report") == "1"
    assert value("--steps-per-eval") == "2"
    assert value("--save-every") == "3"
    assert value("--seed") == "11"
    assert value("--fine-tune-type") == "lora"
    # SFT trains on the completion only; a corpus must not ask for prompt masking
    assert "--mask-prompt" in cmd
    assert "--mask-prompt" not in _lora_command(
        "base", tmp_path / "data", tmp_path / "adapter", options, "text"
    )


def test_adapter_weights_prefers_the_final_file_and_takes_the_newest(tmp_path):
    assert _adapter_weights(tmp_path) is None
    (tmp_path / "0000010_adapters.safetensors").write_bytes(b"x")
    (tmp_path / "0000002_adapters.safetensors").write_bytes(b"x")
    (tmp_path / "junk_adapters.safetensors").write_bytes(b"x")  # not a checkpoint
    assert _adapter_weights(tmp_path) == tmp_path / "0000010_adapters.safetensors"
    (tmp_path / "adapters.safetensors").write_bytes(b"x")
    assert _adapter_weights(tmp_path) == tmp_path / "adapters.safetensors"


# ---- perplexity windows (pure, no MLX needed) ------------------------------


def test_segments_score_every_token_after_the_first_exactly_once():
    ids = list(range(1000))
    segments = _segments(ids, max_tokens=4000, window=512)
    scored = [token for segment in segments for token in segment[1:]]
    assert scored == list(range(1, 1000))


def test_segments_truncate_at_max_tokens():
    ids = list(range(1000))
    segments = _segments(ids, max_tokens=100, window=512)
    assert [token for segment in segments for token in segment[1:]] == list(range(1, 100))


# ---- artifact round trip (adapter serialized with base model) --------------


def test_adapter_artifact_round_trip(tmp_path):
    adapter = LoRAAdapter(
        base_model="some/base-model",
        adapter_path=None,  # no trained weights yet: the raw base model
        training={"options": {"iters": 300}},
    )
    saved = adapter.save(tmp_path / "adapter")
    assert (saved / "lora_meta.json").exists()
    loaded = LoRAAdapter.load(saved)
    assert loaded.base_model == "some/base-model"
    assert loaded.training["options"]["iters"] == 300


def test_adapter_without_adapters_loads_fine(tmp_path):
    # a copy of only the meta (adapter weights omitted) still loads as the
    # base model with adapter off: the raw-model comparison
    (tmp_path / "a").mkdir()
    LoRAAdapter(base_model="b/x", adapter_path=None, training={}).save(tmp_path / "a")
    loaded = LoRAAdapter.load(tmp_path / "a")
    assert loaded.adapter_path is None


def test_saving_an_adapter_without_weights_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    adapter = LoRAAdapter(base_model="b/x", adapter_path=str(tmp_path / "empty"), training={})
    with pytest.raises(FileNotFoundError, match=r"adapters\.safetensors"):
        adapter.save(tmp_path / "artifact")


def test_model_import_requires_apple(monkeypatch):
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


def test_echo_inserts_one_space_between_words():
    assert _echo("hello", "world") == "hello world"
    assert _echo("hello ", "world") == "hello world"
    assert _echo("hello", " world") == "hello world"


def test_echo_leaves_empty_sides_and_punctuation_alone():
    assert _echo("", "hello") == "hello"
    assert _echo("hello", "") == "hello"
    assert _echo("hello", ".") == "hello."
    assert _echo("hello", "\nworld") == "hello\nworld"


# ---- end-to-end (converted model required) ---------------------------------


@slow
def test_train_lora_tiny_run(tmp_path):
    """2 iterations on the converted 4-bit model: adapter produced, model
    generates and evaluates, save/load round trip reproduces identical greedy
    output."""
    model_dir = cached_path(BRITLLM_4BIT)
    if model_dir is None:
        pytest.skip(_NO_MODEL)
    data = LoRAData.from_texts([f"the quick brown fox jumped over fence {i}" for i in range(8)])
    adapter = train_lora(
        model_dir,
        data,
        tmp_path / "adapter",
        options=LoRAOptions(iters=2, batch_size=1, max_seq_length=128, num_layers=4),
    )
    assert (Path(adapter.adapter_path) / "adapters.safetensors").exists()
    assert adapter.training["options"]["iters"] == 2

    model = LoRAModel(adapter.base_model, adapter.adapter_path)
    prompt = "the quick brown fox"
    out_on = model.generate(prompt, temperature=0, max_new_tokens=8, seed=0)
    assert isinstance(out_on, str) and out_on.startswith(prompt)
    scores = model.evaluate("the quick brown fox jumped over the fence")
    assert scores["perplexity"] > 0

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


@slow
def test_train_lora_greedy_determinism(tmp_path):
    model_dir = cached_path(BRITLLM_4BIT)
    if model_dir is None:
        pytest.skip(_NO_MODEL)
    data = LoRAData.from_texts(["alpha beta gamma delta." for _ in range(6)])
    a = train_lora(
        model_dir,
        data,
        tmp_path / "a1",
        options=LoRAOptions(iters=2, batch_size=1, max_seq_length=64, num_layers=2, seed=7),
    )
    m1 = LoRAModel(a.base_model, a.adapter_path)
    prompt, kwargs = "alpha", {"temperature": 1.0, "max_new_tokens": 10, "seed": 3}
    first = m1.generate(prompt, **kwargs)
    second = m1.generate(prompt, **kwargs)
    assert first == second  # seeded sampling is deterministic


def test_negative_temperature_raises():
    """Same convention as LanguageModel.generate: 0 is greedy, <0 is an error."""
    model = object.__new__(LoRAModel)
    with pytest.raises(ValueError, match="temperature"):
        model.generate("prompt", temperature=-1.0)
