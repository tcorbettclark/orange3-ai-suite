"""The training child's own logic, without MLX and without a subprocess.

The child is the one place that knows how mlx-lm is called, so it is tested
directly here: the specification it accepts, the metrics it writes, and the two
ways it can be lied to by a future mlx-lm.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from orangecontrib.ai_suite.kernel.lora import _child
from orangecontrib.ai_suite.kernel.lora._protocol import MARKER, decode

SPEC = {
    "model": "the-base",
    "data": "/runs/1/data",
    "adapter_path": "/runs/1/adapter",
    "mask_prompt": True,
    "iters": 7,
    "learning_rate": 2e-5,
    "num_layers": 3,
    "batch_size": 2,
    "max_seq_length": 64,
    "steps_per_report": 1,
    "steps_per_eval": 2,
    "save_every": 3,
    "seed": 11,
}


class _TrainingCallback:
    """Stands in for mlx_lm.tuner.callbacks.TrainingCallback."""

    def on_train_loss_report(self, info):
        raise AssertionError("the base class is not called")

    def on_val_loss_report(self, info):
        raise AssertionError("the base class is not called")


def _lines(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [decode(line + "\n") for line in out.splitlines() if line]


# ---- the specification -----------------------------------------------------


def test_the_spec_fills_the_namespace_mlx_lm_reads():
    args = SimpleNamespace(trust_remote_code=False, save_every=100)
    _child.apply_spec(args, SPEC)
    assert args.model == "the-base"
    assert args.data == "/runs/1/data"
    assert args.adapter_path == "/runs/1/adapter"
    assert args.iters == 7
    assert args.learning_rate == 2e-5
    assert args.save_every == 3  # overrides mlx-lm's default
    assert args.seed == 11
    assert args.train is True and args.test is False
    assert args.fine_tune_type == "lora"
    assert args.mask_prompt is True
    assert args.trust_remote_code is False  # mlx-lm's default is left alone


def test_a_text_run_does_not_mask_the_prompt():
    args = SimpleNamespace(trust_remote_code=False)
    _child.apply_spec(args, {**SPEC, "mask_prompt": False})
    assert args.mask_prompt is False


@pytest.mark.parametrize("name", sorted(SPEC))
def test_a_missing_setting_is_refused(name):
    args = SimpleNamespace(trust_remote_code=False)
    spec = {key: value for key, value in SPEC.items() if key != name}
    with pytest.raises(KeyError, match="missing"):
        _child.apply_spec(args, spec)


def test_an_unknown_setting_is_refused():
    # The parent derives the spec from LoRAOptions; a key the child cannot apply
    # means a setting was added and would otherwise be dropped in silence.
    args = SimpleNamespace(trust_remote_code=False)
    with pytest.raises(KeyError, match="mlx-lm cannot use"):
        _child.apply_spec(args, {**SPEC, "grad_checkpoint": True})


# ---- the metrics -----------------------------------------------------------


def test_a_training_report_becomes_the_widgets_metric(capsys):
    emitter = _child.emit_metrics(_TrainingCallback)
    emitter.on_train_loss_report(
        {"iteration": 20, "train_loss": 0.725, "tokens_per_second": 184.0, "peak_memory": 3.2}
    )
    assert _lines(capsys) == [{"kind": "train", "iter": 20, "loss": 0.725, "tok_s": 184}]


def test_a_validation_report_is_put_on_the_training_axis(capsys):
    # mlx-lm reports the iteration *before* the evaluation ("iteration": it - 1),
    # so the child's +1 is what makes the two curves line up in the widget.
    emitter = _child.emit_metrics(_TrainingCallback)
    emitter.on_val_loss_report({"iteration": 0, "val_loss": 3.419, "val_time": 0.02})
    assert _lines(capsys) == [{"kind": "val", "iter": 1, "loss": 3.419}]


def test_the_metrics_are_json_lines_with_the_shared_marker(capsys):
    emitter = _child.emit_metrics(_TrainingCallback)
    emitter.on_train_loss_report({"iteration": 1, "train_loss": 1.0, "tokens_per_second": 1.0})
    out = capsys.readouterr().out
    assert out.startswith(MARKER)
    assert out.endswith("\n")


# ---- the trainable-parameter hook ------------------------------------------


def test_the_params_hook_keeps_the_human_line_and_emits_the_numbers(monkeypatch):
    printed, emitted = [], []
    stand_in = SimpleNamespace(print_trainable_parameters=printed.append)
    monkeypatch.setattr(_child, "emit_params", emitted.append)

    _child.install_params_hook(stand_in)
    stand_in.print_trainable_parameters("the model")

    assert printed == ["the model"]  # mlx-lm's own line survives
    assert emitted == ["the model"]  # ... and the counts become data


def test_the_parameter_counts_are_mlx_lm_arithmetic(monkeypatch, capsys):
    # A fake mlx/mlx_lm pair: the child's arithmetic must be the same sum mlx-lm
    # prints, so the metric and the log line agree.
    mlx_utils = types.ModuleType("mlx.utils")
    mlx_utils.tree_flatten = lambda tree: list(tree)
    lm_utils = types.ModuleType("mlx_lm.utils")
    lm_utils.get_total_parameters = lambda model: 2_667_000_000.0
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm_utils)

    class _Leaf:
        def __init__(self, size):
            self.size = size

    model = SimpleNamespace(
        trainable_parameters=lambda: [("lora_a", _Leaf(6_210_000.0)), ("lora_b", _Leaf(0.0))]
    )
    _child.emit_params(model)
    metric = _lines(capsys)[0]
    assert metric["kind"] == "params"
    assert metric["pct"] == pytest.approx(100 * 6_210_000 / 2_667_000_000)
    assert metric["detail"] == "6.210M/2667.000M"


def test_a_moved_printer_fails_loudly():
    # If mlx-lm stops going through that module global, the metric would vanish;
    # the child says so instead, at the start of the run.
    with pytest.raises(RuntimeError, match="print_trainable_parameters"):
        _child.install_params_hook(SimpleNamespace())
