"""``orangecontrib.ai_suite.kernel.lora``: the data, artifact, protocol and training tests.

Data, artifact and protocol tests always run; the end-to-end training tests need the
converted ``britllm-4bit`` model on disk and skip without it. No test here imports
Orange or Qt.
"""

import dataclasses
import json
import sys
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
    mlx,
    split_for_training,
    train_lora,
)
from orangecontrib.ai_suite.kernel.lora.mlx.protocol import (
    MARKER,
    Params,
    TrainReport,
    ValReport,
    decode,
    encode,
)
from orangecontrib.ai_suite.kernel.lora.mlx.runtime import segments as _segments
from orangecontrib.ai_suite.kernel.lora.mlx.train import command as child_command
from orangecontrib.ai_suite.kernel.lora.model import _echo
from orangecontrib.ai_suite.kernel.lora.training import _run_spec

_NO_MODEL = (
    "pinned 4-bit model not cached — use the BritLLM Models widget or "
    "`orange-ai-assets fetch britllm-4bit`"
)
slow = pytest.mark.slow  # end-to-end runs (model load + iterations); excluded
# from the default suite via addopts -m "not slow"; run with `-m slow` when
# the converted model is present to validate the full path in one go.

# ---- data staging (the two teaching formats) ------------------------------


def test_text_corpus_staging(tmp_path):
    data = LoRAData.from_texts([f"document {i} text" for i in range(20)])
    assert data.format == "text"
    assert len(data.rows) == 20
    train, valid = data.split(valid_fraction=0.1, seed=0)
    assert len(train) == 18 and len(valid) == 2  # 10% -> val
    staged = mlx.write_dataset(tmp_path / "data", data.format, train, valid)
    train_rows = [json.loads(line) for line in (staged / "train.jsonl").read_text().splitlines()]
    valid_rows = [json.loads(line) for line in (staged / "valid.jsonl").read_text().splitlines()]
    assert len(train_rows) + len(valid_rows) == 20
    assert all(set(r) == {"text"} for r in train_rows + valid_rows)


def test_pair_corpus_staging_is_sft(tmp_path):
    pairs = [(f"question {i}?", f"answer {i}.") for i in range(10)]
    data = LoRAData.from_pairs(pairs)
    assert data.format == "pairs"
    train, valid = data.split(valid_fraction=0.1, seed=0)
    staged = mlx.write_dataset(tmp_path / "data", data.format, train, valid)
    rows = [
        json.loads(line)
        for name in ("train", "valid")
        for line in (staged / f"{name}.jsonl").read_text().splitlines()
    ]
    assert all(set(r) == {"prompt", "completion"} for r in rows)
    # the SFT lesson: prompt and completion are separately authored, and the
    # staged rows are exactly the pairs that went in
    assert sorted((r["prompt"], r["completion"]) for r in rows) == sorted(pairs)


def test_the_split_reads_positions_not_content():
    texts = [f"document {i}" for i in range(40)]
    first_train, first_valid = LoRAData.from_texts(texts).split(valid_fraction=0.25, seed=3)
    # every 4th row, offset by the seed: 3, 7, 11, ... 39
    assert first_valid == [texts[i] for i in range(3, 40, 4)]
    assert first_train == [row for i, row in enumerate(texts) if i % 4 != 3]

    # Reordering the corpus moves rows across the split, and identical rows are
    # no exception: the contract is about where a row sits, not what it says.
    _reversed_train, reversed_valid = LoRAData.from_texts(list(reversed(texts))).split(
        valid_fraction=0.25, seed=3
    )
    assert set(reversed_valid) != set(first_valid)
    duplicated = LoRAData.from_texts(["same"] * 8).split(valid_fraction=0.25, seed=0)
    assert duplicated[1] == ["same"] * 2  # two of eight, both identical


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
    staged = mlx.write_dataset(tmp_path / "data", "text", ["one", "two"], [])
    assert (staged / "train.jsonl").exists()
    assert not (staged / "valid.jsonl").exists()
    # a stale file from an earlier run must not be picked up
    (staged / "valid.jsonl").write_text("stale\n")
    mlx.write_dataset(staged, "text", ["one", "two"], [])
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
    with pytest.raises(ValueError, match="seed"):
        LoRAOptions(seed=-1)  # mlx-lm's --seed is unsigned
    with pytest.raises(FrozenInstanceError):
        LoRAOptions().iters = 1


# ---- the child's wire format (the widget's live-loss source) --------------


@pytest.mark.parametrize(
    "event",
    [
        Params(fraction=0.233, detail="6.210M/2667.000M"),
        TrainReport(iteration=20, loss=0.725, tokens_per_second=184),
        ValReport(iteration=1, loss=3.419),
    ],
)
def test_an_event_round_trips_through_the_wire_format(event):
    assert decode(encode(event) + "\n") == event


def test_every_event_type_has_a_wire_name():
    from orangecontrib.ai_suite.kernel.lora.mlx.protocol import _BY_KIND, _KINDS

    assert set(_KINDS) == {Params, TrainReport, ValReport}
    assert set(_BY_KIND) == {"params", "train", "val"}


def test_mlx_lm_human_output_is_not_a_metric():
    # Everything the child does not mark stays a log line, ANSI and all.
    for line in (
        "Loading pretrained model\n",
        "Trainable parameters: 0.233% (6.210M/2667.000M)\n",
        "\x1b[1miter   train_loss     tok/s\x1b[0m\n",
        "\n",
        "",
    ):
        assert decode(line) is None


def test_a_malformed_event_is_a_bug_in_our_own_child():
    with pytest.raises(ValueError, match="malformed event"):
        decode(MARKER + "{not json}\n")


def test_an_unknown_event_kind_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown event kind"):
        decode(MARKER + '{"kind": "epoch", "loss": 1.0}\n')


# ---- what the child is asked for -------------------------------------------


def test_the_child_spec_carries_every_option_but_the_split(tmp_path):
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
    spec = _run_spec("base", tmp_path / "data", tmp_path / "adapter", options, "pairs")
    assert spec.model == "base"
    assert spec.data == str(tmp_path / "data")
    assert spec.adapter_path == str(tmp_path / "adapter")
    assert spec.iters == 7
    assert spec.learning_rate == 2e-5
    assert spec.num_layers == 3
    assert spec.batch_size == 2
    assert spec.max_seq_length == 64
    assert spec.steps_per_report == 1
    assert spec.steps_per_eval == 2
    assert spec.save_every == 3
    assert spec.seed == 11
    assert spec.mask_prompt is True  # SFT: the prompt is not the target
    # every setting the run owns travels: the two vocabularies agree field for
    # field, bar the fraction, which decides the staging and never reaches mlx-lm
    carried = {field.name for field in dataclasses.fields(spec)}
    options = {field.name for field in dataclasses.fields(LoRAOptions)}
    assert carried == (options - {"validation_fraction"}) | {
        "model",
        "data",
        "adapter_path",
        "mask_prompt",
    }


def test_a_text_run_does_not_mask_the_prompt(tmp_path):
    spec = _run_spec("base", tmp_path, tmp_path, LoRAOptions(), "text")
    assert spec.mask_prompt is False


def test_the_command_runs_the_boundary_child_with_the_spec_as_one_argument():
    # The child module lives inside the boundary package: nothing above it needs
    # to know that training is a process at all.
    spec = _run_spec("base", "/data", "/adapter", LoRAOptions(iters=1), "text")
    cmd = child_command(spec)
    assert cmd[:3] == [sys.executable, "-m", "orangecontrib.ai_suite.kernel.lora.mlx.child"]
    payload = json.loads(cmd[3])
    assert payload["iters"] == 1
    assert set(payload) == {field.name for field in dataclasses.fields(spec)}


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
    """A non-Apple machine gets a clear message, never a stack trace."""
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
    """Two iterations on the converted 4-bit model, from training to a reloaded artifact."""
    model_dir = cached_path(BRITLLM_4BIT)
    if model_dir is None:
        pytest.skip(_NO_MODEL)
    data = LoRAData.from_texts([f"the quick brown fox jumped over fence {i}" for i in range(8)])
    training = train_lora(
        model_dir,
        data,
        options=LoRAOptions(iters=2, batch_size=1, max_seq_length=128, num_layers=4),
    )
    events = list(training)
    assert any(isinstance(event, TrainReport) for event in events)
    adapter = training.adapter
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
    training = train_lora(
        model_dir,
        data,
        options=LoRAOptions(iters=2, batch_size=1, max_seq_length=64, num_layers=2, seed=7),
    )
    list(training)
    a = training.adapter
    m1 = LoRAModel(a.base_model, a.adapter_path)
    prompt, kwargs = "alpha", {"temperature": 1.0, "max_new_tokens": 10, "seed": 3}
    first = m1.generate(prompt, **kwargs)
    second = m1.generate(prompt, **kwargs)
    assert first == second  # seeded sampling is deterministic


def test_negative_temperature_raises():
    """Temperature 0 is greedy and a negative one is an error, as in ``LanguageModel.generate``."""
    model = object.__new__(LoRAModel)
    with pytest.raises(ValueError, match="temperature"):
        model.generate("prompt", temperature=-1.0)
