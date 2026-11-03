"""LM Trainer widget: settings plumbing, threading path, graceful degradation.

The full-training test only runs when Keras is importable in *this* Orange
environment (the [lm] extra). Without it, the degradation-path test asserts
the clear 'install' message instead of a crash (spec §3).
"""

import importlib.util

import numpy as np
import pytest
from Orange.data import Domain, ContinuousVariable, Table

from orangecontrib.nn_suite.adapters import checkpoint_table, table_texts
from orangecontrib.nn_suite.widgets.ow_lm_trainer import OWLMTrainer, TrainingWorker

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None


def _numbers_table():
    dom = Domain([ContinuousVariable("x")])
    return Table.from_numpy(dom, np.arange(8.0).reshape(-1, 1))


def test_table_texts_adapter(text_table, texts):
    assert table_texts(text_table) == texts


def test_textless_data_sets_warning(qapp):
    w = OWLMTrainer()
    w.show()
    w.set_data(_numbers_table())
    assert w.Warning.no_text.is_shown()
    w.start_training()  # must not crash, must not spawn a worker
    assert w.worker is None


def test_trainer_outputs_commit_on_done(qapp, capture, text_table):
    """Full pipeline in-thread (blocking): needs Keras."""
    if not KERAS_AVAILABLE:
        pytest.skip("Keras not installed in this Orange environment ([lm] extra)")
    w = OWLMTrainer()
    w.show()
    w.set_data(text_table)
    w.epochs = 2
    w.context_size = 8
    w.embedding_dim = 8
    w.hidden_units = 16
    w.sample_max_new_tokens = 8
    model_sent = capture(w.Outputs.model)
    checkpoint_sent = capture(w.Outputs.checkpoint)
    w.run_training_blocking()  # same-thread: signals are direct, tests stay simple
    assert w.model is not None
    assert model_sent[-1] is w.model
    cp = checkpoint_sent[-1]
    assert len(cp) == 2  # one row per epoch
    assert cp.domain[0].name == "epoch" and cp.domain.metas[0].name == "sample"
    # the sample column records the gibberish->pastiche progression
    assert all(str(s) for s in cp.metas[:, 0])
    losses = [row["loss"] for row in w.epoch_stats]
    if len(losses) > 1:
        assert losses[-1] < losses[0]


def test_worker_stop_flag_honoured(qapp, capture, text_table):
    if not KERAS_AVAILABLE:
        pytest.skip("Keras not installed in this Orange environment ([lm] extra)")
    w = OWLMTrainer()
    w.show()
    w.set_data(text_table)
    w.epochs = 200  # would take a while; the stop flag must cut it short
    w.context_size = 8
    w.embedding_dim = 8
    w.hidden_units = 16
    w.worker = None
    model_sent = capture(w.Outputs.model)
    w.run_training_blocking()
    # the worker stopped early: far fewer stats recorded than requested epochs
    assert 0 < len(w.epoch_stats) < 200
    assert w.model is not None and model_sent[-1] is w.model


def test_stop_button_sets_flag(qapp, text_table):
    w = OWLMTrainer()
    w.show()
    w.set_data(text_table)
    w.worker = TrainingWorker([], None, {})
    w.stop_training()
    assert w.worker.request_stop is True


def test_missing_keras_message_routing(qapp):
    w = OWLMTrainer()
    w.show()
    w._on_failed("ModuleNotFoundError: No module named 'keras'")
    assert w.Error.keras_missing.is_shown()
    assert not w.Error.training_failed.is_shown()

    w._on_failed("RuntimeError: kapow")
    assert w.Error.training_failed.is_shown()
    assert not w.Information.training.is_shown()


@pytest.mark.skipif(
    KERAS_AVAILABLE,
    reason="exercises the degradation path, which only exists without Keras",
)
def test_start_training_without_keras_shows_install_hint(qapp, text_table):
    """The spec's graceful-degradation contract: no crash, a clear hint.

    Uses the blocking variant: same lazy-import path as the threaded one,
    without a live QThread (offscreen teardown of threads is fragile).
    """
    w = OWLMTrainer()
    w.show()
    w.set_data(text_table)
    w.run_training_blocking()
    assert w.Error.keras_missing.is_shown()
    assert not w.Information.training.is_shown()


@pytest.mark.skipif(
    not KERAS_AVAILABLE,
    reason="needs Keras to patch the real kernel path (runs once [lm] is installed)",
)
def test_training_failure_surfaces_as_message(qapp, text_table, monkeypatch):
    w = OWLMTrainer()
    w.show()
    w.set_data(text_table)

    def failing_train(*args, **kwargs):
        raise RuntimeError("kapow")

    monkeypatch.setattr("nn_kernel.lm.train_language_model", failing_train)
    w.epochs = 2
    w.run_training_blocking()
    assert w.Error.training_failed.is_shown()
    assert not w.Information.training.is_shown()