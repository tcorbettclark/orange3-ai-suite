"""train_lora against a stand-in mlx_lm child: flags, streaming, stop, failure.

The stand-in is a real subprocess, so Popen, the reader thread, polling and the
process-group signal all run; it needs no MLX and no model weights.
"""

import json
import os
import threading

import pytest

from orangecontrib.ai_suite.kernel.lora import (
    LoRAData,
    LoRAOptions,
    TrainingStopped,
    train_lora,
)

FAKE_MLX_LM = r'''
import argparse, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--model")
parser.add_argument("--train", action="store_true")
parser.add_argument("--data")
parser.add_argument("--adapter-path")
parser.add_argument("--iters", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--num-layers", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--max-seq-length", type=int)
parser.add_argument("--steps-per-report", type=int)
parser.add_argument("--steps-per-eval", type=int)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--seed", type=int)
parser.add_argument("--fine-tune-type")
parser.add_argument("--mask-prompt", action="store_true")
args = parser.parse_args()

if os.environ.get("FAKE_MLX_LM_EXIT"):
    print("mlx_lm: simulated failure", flush=True)
    sys.exit(3)

out = Path(args.adapter_path)
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(sys.argv[1:]))
(out / "adapter_config.json").write_text(
    json.dumps({"fine_tune_type": args.fine_tune_type, "mask_prompt": args.mask_prompt})
)
print("Trainable parameters: 0.233% (6.210M/2667.000M)", flush=True)
for it in range(1, min(args.iters, 3) + 1):
    if it % args.save_every == 0:
        (out / f"{it:07d}_adapters.safetensors").write_bytes(b"weights")
    print(f"  {it:>4}    {1.0 / it:>5.3f} \u25bc      184      3.3k", flush=True)
    print(f"  {it:>4}    val {1.0 / it:>5.3f}    0.02s", flush=True)
    time.sleep(0.05)
    if args.iters > 3:
        time.sleep(5)  # a long run: stay alive until the test stops us
(out / "adapters.safetensors").write_bytes(b"weights")
'''


@pytest.fixture
def fake_mlx_lm(tmp_path, monkeypatch):
    site = tmp_path / "site"
    package = site / "mlx_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(FAKE_MLX_LM)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", str(site) if not existing else f"{site}{os.pathsep}{existing}"
    )
    return site


def _data():
    return LoRAData.from_texts(["one", "two", "three", "four"])


def _options(**kwargs):
    values = dict(iters=3, batch_size=1, validation_fraction=0.0)
    values.update(kwargs)
    return LoRAOptions(**values)


def test_completed_run_streams_metrics_and_returns_the_artifact(fake_mlx_lm, tmp_path):
    seen = []
    adapter = train_lora(
        "base-model",
        _data(),
        tmp_path / "adapter",
        options=_options(iters=3),
        on_metric=seen.append,
    )
    assert (tmp_path / "adapter" / "adapters.safetensors").exists()
    assert adapter.hyperparameters["iters_completed"] == 3
    assert adapter.hyperparameters["stopped"] is False
    assert {m["kind"] for m in seen} == {"params", "train", "val"}
    config = adapter.hyperparameters["adapter_config"]
    assert config["fine_tune_type"] == "lora"
    assert config["mask_prompt"] is False
    argv = json.loads((tmp_path / "adapter" / "argv.json").read_text())
    assert argv[0] == "lora"
    assert "--save-every" in argv
    assert "--mask-prompt" not in argv


def test_a_pairs_run_masks_the_prompt_in_the_child(fake_mlx_lm, tmp_path):
    adapter = train_lora(
        "base-model",
        LoRAData.from_pairs([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]),
        tmp_path / "adapter",
        options=_options(iters=1),
    )
    assert adapter.hyperparameters["adapter_config"]["mask_prompt"] is True


def test_stop_adopts_the_last_checkpoint(fake_mlx_lm, tmp_path):
    adapter_dir = tmp_path / "adapter"
    reported = threading.Event()

    def on_metric(metric):
        if metric["kind"] == "train":
            reported.set()

    adapter = train_lora(
        "base-model",
        _data(),
        adapter_dir,
        options=_options(iters=1000, save_every=1),
        on_metric=on_metric,
        stop_requested=reported.is_set,
    )
    assert (adapter_dir / "adapters.safetensors").exists()
    assert adapter.hyperparameters["iters_completed"] == 1
    assert adapter.hyperparameters["stopped"] is True


def test_stop_before_the_first_checkpoint_raises(fake_mlx_lm, tmp_path):
    reported = threading.Event()

    def on_metric(metric):
        if metric["kind"] == "train":
            reported.set()

    with pytest.raises(TrainingStopped, match="stopped before"):
        train_lora(
            "base-model",
            _data(),
            tmp_path / "adapter",
            options=_options(iters=1000, save_every=100),
            on_metric=on_metric,
            stop_requested=reported.is_set,
        )


def test_a_nonzero_exit_reports_the_tail(fake_mlx_lm, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MLX_LM_EXIT", "1")
    with pytest.raises(RuntimeError, match=r"failed \(3\)") as excinfo:
        train_lora("base-model", _data(), tmp_path / "adapter", options=_options())
    assert "simulated failure" in str(excinfo.value)
