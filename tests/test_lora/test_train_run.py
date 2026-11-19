"""``train_lora`` against a stand-in mlx_lm: metrics, checkpoints, stop, failure.

The stand-in is a real subprocess running our child module, so Popen, the reader
thread, polling and the process-group signal all run; only mlx-lm itself is replaced.
It stands in for the *Python API* the child calls: ``build_parser``, ``load``,
``load_dataset``, ``train_model``, ``print_trainable_parameters`` and
``TrainingCallback``.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from orangecontrib.ai_suite.kernel.lora import (
    LoRAAdapter,
    LoRAData,
    LoRAOptions,
    Params,
    TrainingStopped,
    TrainReport,
    ValReport,
    train_lora,
)

#: The stand-in packages: `mlx/__init__.py`, `mlx/utils.py`, `mlx_lm/__init__.py`
FAKE_MLX_INIT = ""
#: `mlx.utils.tree_flatten` - the only MLX function our child uses.
FAKE_MLX_UTILS = r"""
def tree_flatten(tree):
    return list(tree)
"""
FAKE_MLX_LM_INIT = r"""
class _Leaf:
    def __init__(self, size):
        self.size = size


class _Model:
    # Enough of an MLX module for the trainable-parameter report.

    def trainable_parameters(self):
        return [("lora_a", _Leaf(6_210_000.0)), ("lora_b", _Leaf(0.0))]


def load(model, tokenizer_config=None, trust_remote_code=None):
    return _Model(), object()
"""

#: `mlx_lm/utils.py` - the parameter counter our child uses.
FAKE_MLX_LM_UTILS = r"""
def get_total_parameters(model):
    return 2_667_000_000.0
"""

#: `mlx_lm/lora.py` - the names our child reaches for, nothing else.
FAKE_MLX_LM_LORA = r"""
import json, os, sys, time
from pathlib import Path


class _Args:
    # Only the defaults our child relies on, spelled out so a rename shows up.
    model = None
    data = None
    adapter_path = None
    train = False
    test = False
    iters = 100
    learning_rate = 1e-5
    num_layers = 8
    batch_size = 4
    max_seq_length = 2048
    steps_per_report = 10
    steps_per_eval = 200
    save_every = 100
    seed = 0
    fine_tune_type = "lora"
    mask_prompt = None
    trust_remote_code = False


class _Parser:
    def parse_args(self, argv):
        return _Args()


#: `mlx_lm.lora.CONFIG_DEFAULTS` - applied by mlx-lm's `main`, not by the parser, so a
#: namespace just built lacks the three it declares no flag for and carries the rest
#: as None. Both shapes are read by `train_model`, and both are in here.
CONFIG_DEFAULTS = {
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "optimizer_config": {"adam": {}},
    "lr_schedule": None,
    "val_batches": 25,
}


def build_parser():
    return _Parser()


def load_dataset(args, tokenizer):
    return ["train"], ["valid"], []


def print_trainable_parameters(model):
    print("Trainable parameters: 0.233% (6.210M/2667.000M)", flush=True)


def train_model(args, model, train_set, valid_set, training_callback=None):
    # What the real train_model reads first: a setting its parser has no attribute for,
    # and one the parser leaves as None. A child that skips CONFIG_DEFAULTS dies here.
    print(f"lora rank {args.lora_parameters['rank']}", flush=True)
    if args.val_batches is None:
        raise SystemExit("mlx_lm.train_model needs args.val_batches")
    if os.environ.get("FAKE_MLX_LM_EXIT"):
        print("mlx_lm: simulated failure", flush=True)
        sys.exit(3)

    out = Path(args.adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "child.pid").write_text(str(os.getpid()))  # for the reaping test
    # what mlx-lm's save_config(vars(args), ...) writes, and the run as asked for
    (out / "adapter_config.json").write_text(json.dumps(vars(args), default=str))
    (out / "asked.json").write_text(json.dumps(vars(args), default=str))

    print_trainable_parameters(model)
    for it in range(1, min(args.iters, 3) + 1):
        if it % args.save_every == 0:
            # mlx-lm writes the unnumbered adapter on every save and the numbered one
            # beside it, so a run mid-flight has both; the config is one per run,
            # written once before training starts.
            (out / "adapters.safetensors").write_bytes(b"weights")
            (out / f"{it:07d}_adapters.safetensors").write_bytes(b"weights")
        training_callback.on_train_loss_report(
            {"iteration": it, "train_loss": 1.0 / it, "tokens_per_second": 184.0}
        )
        training_callback.on_val_loss_report(
            {"iteration": it - 1, "val_loss": 2.0 / it, "val_time": 0.02}
        )
        time.sleep(0.05)
        if args.iters > 3:
            time.sleep(5)  # a long run: stay alive until the test stops us
    (out / "adapters.safetensors").write_bytes(b"weights")
"""

#: `mlx_lm/tuner/callbacks.py` - the base class our child subclasses.
FAKE_MLX_LM_CALLBACKS = r"""
class TrainingCallback:
    def on_train_loss_report(self, info):
        pass

    def on_val_loss_report(self, info):
        pass
"""


@pytest.fixture
def fake_mlx_lm(tmp_path, monkeypatch):
    site = tmp_path / "site"
    mlx = site / "mlx"
    package = site / "mlx_lm"
    tuner = package / "tuner"
    for directory in (mlx, package, tuner):
        directory.mkdir(parents=True)
    (mlx / "__init__.py").write_text(FAKE_MLX_INIT)
    (mlx / "utils.py").write_text(FAKE_MLX_UTILS)
    (package / "__init__.py").write_text(FAKE_MLX_LM_INIT)
    (package / "lora.py").write_text(FAKE_MLX_LM_LORA)
    (package / "utils.py").write_text(FAKE_MLX_LM_UTILS)
    (tuner / "__init__.py").write_text("")
    (tuner / "callbacks.py").write_text(FAKE_MLX_LM_CALLBACKS)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(site) if not existing else f"{site}{os.pathsep}{existing}")
    return site


def _data():
    return LoRAData.from_texts(["one", "two", "three", "four"])


def _options(**kwargs):
    values = dict(iters=3, batch_size=1, validation_fraction=0.0)
    values.update(kwargs)
    return LoRAOptions(**values)


def _run(**kwargs):
    """Drive a run to the end and give back its events and its artifact."""
    training = train_lora(**kwargs)
    events = list(training)
    return events, training.adapter


def _stop_at_first_report(training):
    """Iterate until the first training report, then ask the run to stop."""
    for event in training:
        if isinstance(event, TrainReport):
            training.stop()


def test_a_completed_run_reports_its_events_and_yields_the_artifact(fake_mlx_lm, tmp_path):
    seen, adapter = _run(
        base_model="base-model",
        data=_data(),
        options=_options(iters=3),
        workdir=tmp_path,
    )
    adapter_dir = Path(adapter.adapter_path)
    assert (adapter_dir / "adapters.safetensors").exists()
    # the run owns fresh scratch: the adapter is a sibling of its staged data
    assert adapter_dir.parent.name.startswith("run-")  # inside the caller's workdir
    assert (adapter_dir.parent / "data" / "train.jsonl").is_file()
    assert adapter.training["run_dir"] == str(adapter_dir.parent)
    assert adapter.training["iters_completed"] == 3
    assert adapter.training["stopped"] is False
    assert {type(event) for event in seen} == {Params, TrainReport, ValReport}
    config = adapter.training["adapter_config"]
    assert config["fine_tune_type"] == "lora"
    assert config["mask_prompt"] is False

    # What mlx-lm was actually asked for: the run's options, not flags.
    asked = json.loads((adapter_dir / "asked.json").read_text())
    assert asked["train"] is True and asked["test"] is False
    assert asked["model"] == "base-model"
    assert asked["iters"] == 3
    assert asked["save_every"] == 50
    assert asked["fine_tune_type"] == "lora"
    # mlx-lm's own defaults too, which its parser does not carry: run's spec first,
    # then whatever the run did not ask about.
    assert asked["lora_parameters"]["rank"] == 8
    assert asked["val_batches"] == 25

    # One event per report, in the order they happened, typed rather than
    # tagged: the params report arrives once, before any loss.
    assert isinstance(seen[0], Params)
    train = [event for event in seen if isinstance(event, TrainReport)]
    val = [event for event in seen if isinstance(event, ValReport)]
    assert [event.iteration for event in train] == [1, 2, 3]
    assert [event.iteration for event in val] == [1, 2, 3]  # val shares train's axis
    assert train[0].loss == pytest.approx(1.0)
    assert train[0].tokens_per_second == 184
    assert seen[0].fraction == pytest.approx(100 * 6_210_000 / 2_667_000_000)
    assert seen[0].detail == "6.210M/2667.000M"


def test_a_pairs_run_masks_the_prompt_in_the_child(fake_mlx_lm, tmp_path):
    _events, adapter = _run(
        base_model="base-model",
        data=LoRAData.from_pairs([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]),
        options=_options(iters=1),
        workdir=tmp_path,
    )
    assert adapter.training["adapter_config"]["mask_prompt"] is True
    asked = json.loads((Path(adapter.adapter_path) / "asked.json").read_text())
    assert asked["mask_prompt"] is True


def test_a_finished_run_that_checkpointed_is_not_read_as_stopped(fake_mlx_lm, tmp_path):
    # The unnumbered adapter is written on every save, not only at the end, so its
    # presence cannot be what decides that a run finished: iters_completed would be
    # the iterations asked for rather than the ones reached.
    _events, adapter = _run(
        base_model="base-model",
        data=_data(),
        options=_options(iters=3, save_every=1),
        workdir=tmp_path,
    )
    assert adapter.training["iters_completed"] == 3
    assert adapter.training["stopped"] is False
    assert adapter.training["adapter_config"]["iters"] == 3


def test_two_runs_never_share_a_directory(fake_mlx_lm, tmp_path):
    _events, first = _run(
        base_model="base-model", data=_data(), options=_options(iters=1), workdir=tmp_path
    )
    _events, second = _run(
        base_model="base-model", data=_data(), options=_options(iters=1), workdir=tmp_path
    )
    assert first.adapter_path != second.adapter_path  # each run its own scratch


def test_stop_adopts_the_last_checkpoint(fake_mlx_lm, tmp_path):
    training = train_lora(
        "base-model",
        _data(),
        options=_options(iters=1000, save_every=1),
        workdir=tmp_path,
    )
    _stop_at_first_report(training)
    adapter = training.adapter
    adapter_dir = Path(adapter.adapter_path)
    assert (adapter_dir / "adapters.safetensors").exists()
    assert adapter.training["iters_completed"] == 1
    assert adapter.training["stopped"] is True

    # The adopted checkpoint keeps the run's config, because that is the only
    # config mlx-lm writes (once, before training). How far the run got is
    # recorded by iters_completed, not by the config.
    config = adapter.training["adapter_config"]
    assert config["fine_tune_type"] == "lora"
    assert config["iters"] == 1000
    assert not list(adapter_dir.glob("*_adapter_config.json")), "mlx-lm writes none"


def test_a_stopped_run_reaps_its_child_before_it_reads_the_directory(fake_mlx_lm, tmp_path):
    # The child saves its checkpoint straight to its final path, so a directory
    # read while it is still alive can pick up a truncated file. By the time the
    # iteration ends, the pid must be gone - exited, and reaped by us.
    training = train_lora(
        "base-model",
        _data(),
        options=_options(iters=1000, save_every=1),
        workdir=tmp_path,
    )
    _stop_at_first_report(training)
    pid = int((Path(training.adapter.adapter_path) / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_stop_before_the_first_checkpoint_raises(fake_mlx_lm, tmp_path):
    training = train_lora(
        "base-model",
        _data(),
        options=_options(iters=1000, save_every=100),
        workdir=tmp_path,
    )
    with pytest.raises(TrainingStopped, match="stopped before"):
        # The stop lands after the first report, before any checkpoint was saved.
        for event in training:
            if isinstance(event, TrainReport):
                training.stop()


def test_a_nonzero_exit_reports_the_tail(fake_mlx_lm, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MLX_LM_EXIT", "1")
    with pytest.raises(RuntimeError, match=r"failed \(3\)") as excinfo:
        list(train_lora("base-model", _data(), options=_options(), workdir=tmp_path))
    assert "simulated failure" in str(excinfo.value)


# ---- whose directory a run writes into -------------------------------------


def test_a_run_without_a_workdir_lands_in_the_temp_root(fake_mlx_lm):
    _events, adapter = _run(base_model="base-model", data=_data(), options=_options(iters=3))
    run_dir = Path(adapter.training["run_dir"])
    try:
        assert run_dir.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        # the shared root is nobody's, so the name says whose directory it is
        assert run_dir.name.startswith("orange-ai-lora-")
        assert (run_dir / "data" / "train.jsonl").is_file()
        assert (run_dir / "adapter" / "adapters.safetensors").is_file()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)  # no workdir, so we own it


def test_a_workdir_gets_a_fresh_child_per_run(fake_mlx_lm, tmp_path):
    workdir = tmp_path / "runs"

    def one():
        return _run(
            base_model="base-model", data=_data(), options=_options(iters=1), workdir=workdir
        )[1]

    dirs = [Path(adapter.training["run_dir"]) for adapter in (one(), one())]
    assert all(run_dir.parent.resolve() == workdir.resolve() for run_dir in dirs)
    assert dirs[0] != dirs[1]  # never a re-run into the previous run's files
    # inside a directory the caller named, a run directory is just a run
    assert all(run_dir.name.startswith("run-") for run_dir in dirs)
    # ... so the workdir holds exactly this caller's runs, which is what makes
    # "delete the workdir" a complete cleanup
    assert sorted(p.name for p in workdir.iterdir()) == sorted(d.name for d in dirs)


def test_a_workdir_is_created_if_it_is_missing(fake_mlx_lm, tmp_path):
    # Asked for, then started: the directory is made on the first event, so a run
    # that is never iterated leaves nothing behind.
    workdir = tmp_path / "runs" / "today"
    training = train_lora("base-model", _data(), options=_options(iters=1), workdir=workdir)
    assert not workdir.exists()
    list(training)
    assert Path(training.adapter.training["run_dir"]).parent == workdir


def test_a_workdir_that_is_a_file_is_refused_when_the_run_is_asked_for(fake_mlx_lm, tmp_path):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("not a directory")
    with pytest.raises(ValueError, match="workdir must be a directory"):
        train_lora("base-model", _data(), options=_options(iters=1), workdir=not_a_directory)


def test_the_artifact_name_is_the_callers(fake_mlx_lm, tmp_path):
    # A generated directory name is not a name to show a person, so the caller
    # supplies it - and it travels into the artifact the run returns.
    _events, adapter = _run(
        base_model="base-model",
        data=_data(),
        options=_options(iters=1),
        workdir=tmp_path,
        name="shakespeare",
    )
    assert adapter.name == "shakespeare"
    assert LoRAAdapter.load(adapter.save(tmp_path / "artifact")).name == "shakespeare"


def test_an_unnamed_run_is_not_named_after_its_directory(fake_mlx_lm, tmp_path):
    _events, adapter = _run(
        base_model="base-model", data=_data(), options=_options(iters=1), workdir=tmp_path
    )
    assert adapter.name == ""
    assert adapter.name != Path(adapter.training["run_dir"]).name


# ---- the run as an object: stop, close, and iterating once -----------------


def test_asking_for_a_run_starts_nothing(fake_mlx_lm, tmp_path):
    # Lazy until iterated, so a caller that changes its mind costs nothing.
    workdir = tmp_path / "runs"
    training = train_lora("base-model", _data(), options=_options(iters=1), workdir=workdir)
    assert list(workdir.iterdir() if workdir.exists() else []) == []
    assert training.adapter is None


def test_a_stop_before_iteration_starts_nothing_and_says_so(fake_mlx_lm, tmp_path):
    training = train_lora("base-model", _data(), options=_options(iters=1), workdir=tmp_path)
    training.stop()
    with pytest.raises(TrainingStopped, match="stopped before it started"):
        list(training)
    # neither a child nor a run directory: nothing was asked of mlx-lm at all
    assert not list(tmp_path.glob("run-*"))


def test_leaving_the_loop_early_stops_and_reaps_the_child(fake_mlx_lm, tmp_path):
    # A consumer that breaks out must not leave a training process behind: the
    # run's `with` terminates it and waits, rather than hoping for a GC.
    training = train_lora(
        "base-model",
        _data(),
        options=_options(iters=1000, save_every=1),
        workdir=tmp_path,
    )
    with training:
        for event in training:
            if isinstance(event, TrainReport):
                pid = int((Path(training._run.spec.adapter_path) / "child.pid").read_text())
                break
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_run_reports_its_events_once(fake_mlx_lm, tmp_path):
    training = train_lora("base-model", _data(), options=_options(iters=1), workdir=tmp_path)
    list(training)
    with pytest.raises(RuntimeError, match="once"):
        list(training)
