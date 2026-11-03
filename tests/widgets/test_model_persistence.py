"""Widget-level model persistence: save/load UI for model artifacts.

One slow test covering the classroom loop: train in LM Trainer → Save model →
load in LM Generator → generate. Plus the LoRA fp16 guard and dispatch checks.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from AnyQt.QtWidgets import QFileDialog
from Orange.data import Domain, StringVariable, Table

from nn_suite.widgets.ow_lm_generator import OWLMGenerator
from nn_suite.widgets.ow_lm_trainer import OWLMTrainer
from nn_suite.widgets.ow_lora_trainer import OWLoRATrainer

KERAS_OK = importlib.util.find_spec("keras") is not None

_TEXTS = ["the cat sat on the mat. the cat saw the dog. a dog and a cat ran."]


def _text_table(texts=_TEXTS):
    domain = Domain([], metas=[StringVariable("text")])
    metas = np.asarray(texts, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(texts), 0)), None, metas)


@pytest.mark.slow
@pytest.mark.skipif(not KERAS_OK, reason="[lm] extra not installed")
def test_train_save_load_generate_round_trip(qapp, capture, monkeypatch, tmp_path):
    """The classroom loop through the widgets' own buttons."""
    # 1. train briefly in LM Trainer
    trainer = OWLMTrainer()
    trainer.show()
    trainer.set_data(_text_table())
    trainer.epochs = 2
    trainer.context_size = 8
    trainer.embedding_dim = 8
    trainer.hidden_units = 16
    trainer.run_training_blocking()
    assert trainer.model is not None

    # 2. Save model (dialog patched)
    model_dir = tmp_path / "gibbon-lm-demo"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(model_dir), "")),
    )
    trainer.save_model()
    assert (model_dir / "meta.json").exists() and (model_dir / "model.keras").exists()
    assert trainer.Information.saved.is_shown()

    # 3. load in LM Generator and generate (dialog patched)
    generator = OWLMGenerator()
    generator.show()
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: str(model_dir)),
    )
    generator.load_model()
    assert generator.Information.loaded.is_shown()
    model = generator.model
    out = model.generate("the cat ", temperature=0, max_new_tokens=8)
    assert out.startswith("the cat ")


def test_generator_load_rejects_plain_directory(qapp, capture, monkeypatch, tmp_path):
    generator = OWLMGenerator()
    generator.show()
    plain = tmp_path / "not-a-model"
    plain.mkdir()
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: str(plain)),
    )
    capture(generator.Outputs.generations)
    generator.load_model()
    assert generator.Warning.not_generative.is_shown()


def test_lora_fp16_guard_blocks_unconverted_model(qapp, text_table, monkeypatch, tmp_path):
    """Spec: 4-bit QLoRA only — an unquantised base model must not train."""
    w = OWLoRATrainer()
    w.show()
    w.set_data(text_table)
    # a local directory WITHOUT quantization in config.json
    unquantised = tmp_path / "fp16-model"
    unquantised.mkdir()
    (unquantised / "config.json").write_text(json.dumps({"model_type": "llama"}))
    w.base_model = str(unquantised)
    w.run_training_blocking()
    assert w.Warning.fp16_model.is_shown()
    assert w.worker is None  # nothing started


def test_lora_fp16_guard_accepts_quantised_dir(qapp, text_table):
    w = OWLoRATrainer()
    quantised_dir = Path(__file__).resolve().parents[2] / "data" / "models" / "britllm-4bit"
    if not (quantised_dir / "config.json").exists():
        pytest.skip("converted model absent")
    assert w._model_is_quantised(str(quantised_dir))
    assert not w._model_is_quantised("britllm/britllm-3b-v0.1")  # HF id: not verifiable


def test_lora_save_adapter(qapp, capture, monkeypatch, tmp_path):
    """Save adapter without training: stub the artifact (kernel-side tested);
    the widget wiring is what's under test here."""
    w = OWLoRATrainer()
    w.show()
    from nn_kernel.lora import LoRAModel

    w.model = LoRAModel("some/base", None)  # raw base artifact shell
    dest = tmp_path / "adapter-out"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")),
    )
    w.save_adapter()
    assert (dest / "lora_meta.json").exists()
    assert w.Information.saved.is_shown()
