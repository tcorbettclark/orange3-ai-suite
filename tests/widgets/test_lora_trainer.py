"""LoRA Trainer widget tests.

The real-training test needs mlx in THIS environment (Apple Silicon [lora]
extra) AND the converted model on disk; deg + routing tests run anywhere.
"""

import importlib.util

import numpy as np
import pytest
from Orange.data import Domain, StringVariable, Table

from nn_suite.widgets.ow_lora_trainer import OWLoRATrainer
from nn_suite.adapters import table_pairs

MLX_OK = importlib.util.find_spec("mlx") is not None
def test_table_pairs_helper():
    domain = Domain([], metas=[
        StringVariable("completion"), StringVariable("prompt"),
        StringVariable("note"),
    ])
    rows = [["answer a", "question a", "x"], ["answer b", "question b", "y"]]
    table = Table.from_numpy(domain, np.empty((2, 0)), None,
                             np.asarray(rows, dtype=object))
    pairs = table_pairs(table)
    assert pairs == [("question a", "answer a"), ("question b", "answer b")]


def test_table_pairs_requires_two_columns():
    domain = Domain([], metas=[StringVariable("only")])
    table = Table.from_numpy(domain, np.empty((1, 0)), None,
                             np.asarray([["x"]], dtype=object).reshape(1, 1))
    with pytest.raises(ValueError, match="two text columns"):
        table_pairs(table)


def test_mlx_missing_routes_to_install_hint(qapp):
    w = OWLoRATrainer()
    w.show()
    w._on_failed("ModuleNotFoundError: No module named 'mlx'")
    assert w.Error.mlx_missing.is_shown()
    w._on_failed("ValueError: something else")
    # mlx message cleared, failure message shown
    assert w.Error.training_failed.is_shown()


@pytest.mark.skipif(
    not MLX_OK, reason="[lora] extra (mlx) not installed in this environment"
)
@pytest.mark.slow
def test_real_training_path(qapp, capture):
    """2-iteration QLoRA via the widget's blocking path end-to-end."""
    from pathlib import Path

    model_file = Path(__file__).resolve().parents[2] / "data" / "models" / "britllm-4bit" / "model.safetensors"
    if not model_file.exists():
        pytest.skip("converted model absent — run demo/fetch_britllm.py")

    w = OWLoRATrainer()
    w.show()
    w.base_model = str(model_file.parent)
    domain = Domain([], metas=[StringVariable("text")])
    texts = [f"the quick brown fox jumped over fence {i}" for i in range(8)]
    table = Table.from_numpy(domain, np.empty((len(texts), 0)), None,
                             np.asarray(texts, dtype=object).reshape(-1, 1))
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