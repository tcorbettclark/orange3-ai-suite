"""LoRA Pre-trainer widget tests.

The real-training test needs mlx in THIS environment (Apple Silicon [lora]
extra) AND the converted model on disk; deg + routing tests run anywhere.
"""

import importlib.util
import json

import numpy as np
import pytest
from Orange.data import Domain, StringVariable, Table

from orangecontrib.ai_suite.widgets.ow_lora_pretrainer import OWLoRAPretrainer

MLX_OK = importlib.util.find_spec("mlx") is not None


def test_pretrainer_builds_text_rows(qapp, text_table):
    w = OWLoRAPretrainer()
    w.set_data(text_table)
    data = w._make_lora_data()
    assert data.kind == "text"
    assert len(data.train) + len(data.valid) == 2


def test_mlx_missing_routes_to_install_hint(qapp):
    w = OWLoRAPretrainer()
    w.show()
    w._on_failed("ModuleNotFoundError: No module named 'mlx'")
    assert w.Error.mlx_missing.is_shown()
    w._on_failed("ValueError: something else")
    # mlx message cleared, failure message shown
    assert w.Error.training_failed.is_shown()


@pytest.mark.skipif(not MLX_OK, reason="[lora] extra (mlx) not installed in this environment")
@pytest.mark.slow
def test_real_training_path(qapp, capture):
    """2-iteration QLoRA via the widget's blocking path end-to-end."""
    from pathlib import Path

    model_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "models"
        / "britllm-4bit"
        / "model.safetensors"
    )
    if not model_file.exists():
        pytest.skip("converted model absent — run demo/fetch_britllm.py")

    w = OWLoRAPretrainer()
    w.show()
    w.base_model = str(model_file.parent)
    domain = Domain([], metas=[StringVariable("text")])
    texts = [f"the quick brown fox jumped over fence {i}" for i in range(8)]
    table = Table.from_numpy(
        domain, np.empty((len(texts), 0)), None, np.asarray(texts, dtype=object).reshape(-1, 1)
    )
    w.set_data(table)
    w.iters = 2
    w.batch_size = 1
    w.max_seq_length = 128
    w.num_layers = 4
    model_sent = capture(w.Outputs.model)
    w.run_training_blocking()
    assert w.model is not None and model_sent[-1] is w.model
    out = w.model.generate("the quick brown", temperature=0, max_new_tokens=6, seed=0)
    assert isinstance(out, str) and out.startswith("the quick brown")


def test_trainable_parameter_note_is_reported(qapp):
    w = OWLoRAPretrainer()
    w.show()
    w._on_metric({"kind": "params", "pct": 0.233, "detail": "6.21M / 2.67B"})
    assert "0.233%" in w.status_label.text()
    assert "6.21M / 2.67B" in w.status_label.text()


def test_lora_fp16_guard_blocks_unconverted_model(qapp, text_table, tmp_path, monkeypatch):
    """4-bit QLoRA only: an unquantised base model must not train."""
    w = OWLoRAPretrainer()
    w.show()
    w.set_data(text_table)
    monkeypatch.setattr(w, "_mlx_available", lambda: True)
    unquantised = tmp_path / "fp16-model"
    unquantised.mkdir()
    (unquantised / "config.json").write_text(json.dumps({"model_type": "llama"}))
    w.base_model = str(unquantised)
    w.run_training_blocking()
    assert w.Warning.fp16_model.is_shown()
    assert w.worker is None


def test_lora_fp16_guard_accepts_quantised_dir(qapp, text_table):
    from pathlib import Path

    w = OWLoRAPretrainer()
    quantised_dir = Path(__file__).resolve().parents[2] / "data" / "models" / "britllm-4bit"
    if not (quantised_dir / "config.json").exists():
        pytest.skip("converted model absent")
    assert w._model_is_quantised(str(quantised_dir))
    assert not w._model_is_quantised("britllm/britllm-3b-v0.1")
