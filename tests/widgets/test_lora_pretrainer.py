"""LoRA Pre-trainer widget tests.

The real-training test needs mlx in THIS environment (Apple Silicon [lora]
extra) AND the converted model on disk; deg + routing tests run anywhere.
"""

import importlib.util

import numpy as np
import pytest
from Orange.data import Domain, StringVariable, Table

from orangecontrib.ai_suite.kernel.assets.britllm import BRITLLM_4BIT
from orangecontrib.ai_suite.kernel.assets.hf import cached_path
from orangecontrib.ai_suite.widgets.ow_lora_pretrainer import OWLoRAPretrainer

MLX_OK = importlib.util.find_spec("mlx") is not None


def test_pretrainer_builds_text_rows(qapp, text_table):
    w = OWLoRAPretrainer()
    w.set_data(text_table)
    data = w._make_lora_data()
    assert data.format == "text"
    assert len(data.rows) == 2


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
    model_dir = cached_path(BRITLLM_4BIT)
    if model_dir is None:
        pytest.skip(
            "pinned 4-bit model not cached — use the BritLLM Models widget "
            "or `orange-ai-assets fetch britllm-4bit`"
        )

    w = OWLoRAPretrainer()
    w.show()
    w.base_model = str(model_dir)
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


def test_stop_before_adapter_routes_to_information(qapp):
    w = OWLoRAPretrainer()
    w.show()
    w._on_stopped("training was stopped before mlx-lm wrote the final adapter")
    assert w.Information.stopped.is_shown()
    assert not w.Information.training.is_shown()


def test_trainable_parameter_note_is_reported(qapp):
    w = OWLoRAPretrainer()
    w.show()
    w._on_metric({"kind": "params", "pct": 0.233, "detail": "6.210M/2667.000M"})
    assert "0.233%" in w.status_label.text()
    assert "6.210M/2667.000M" in w.status_label.text()


def test_lora_without_a_base_model_warns(qapp, text_table, monkeypatch):
    w = OWLoRAPretrainer()
    w.show()
    w.set_data(text_table)
    monkeypatch.setattr(w, "_mlx_available", lambda: True)
    w.base_model = ""
    w.run_training_blocking()
    assert w.Warning.no_model.is_shown()
    assert w.worker is None


def test_small_validation_split_warns_before_starting_a_worker(qapp, text_table, monkeypatch):
    w = OWLoRAPretrainer()
    w.show()
    w.set_data(text_table)
    monkeypatch.setattr(w, "_mlx_available", lambda: True)
    w.base_model = "/tmp/model"
    w.batch_size = 4
    w.validation_fraction = 0.1
    w.run_training_blocking()
    assert w.Warning.bad_settings.is_shown()
    assert w.worker is None


def test_a_valid_split_builds_the_request(qapp, text_table, monkeypatch):
    w = OWLoRAPretrainer()
    w.show()
    w.set_data(text_table)
    monkeypatch.setattr(w, "_mlx_available", lambda: True)
    w.base_model = "/tmp/model"
    w.batch_size = 1
    w.validation_fraction = 0.1
    request = w._build_request()
    assert request is not None
    assert request["options"].validation_fraction == 0.1


def test_base_model_input_sets_the_line_edit_and_the_request(
    qapp, text_table, tmp_path, monkeypatch
):
    w = OWLoRAPretrainer()
    w.show()
    w.set_data(text_table)
    monkeypatch.setattr(w, "_mlx_available", lambda: True)
    model = str(tmp_path / "model")
    w.set_base_model(model)
    assert w.base_model == model
    assert w.base_model_edit.text() == model
    w.batch_size = 1  # the two-row fixture cannot fill a larger validation batch
    request = w._build_request()
    assert request["base_model"] == model
