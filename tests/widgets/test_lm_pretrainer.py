"""LM Pre-trainer widget: settings plumbing, threading path, error reporting."""

import numpy as np
from Orange.data import ContinuousVariable, Domain, Table

from orangecontrib.ai_suite.adapters import table_texts
from orangecontrib.ai_suite.widgets.ow_lm_pretrainer import OWLMPretrainer, TrainingWorker


def _numbers_table():
    dom = Domain([ContinuousVariable("x")])
    return Table.from_numpy(dom, np.arange(8.0).reshape(-1, 1))


def test_table_texts_adapter(text_table, texts):
    assert table_texts(text_table) == texts


def test_textless_data_sets_warning(qapp):
    w = OWLMPretrainer()
    w.show()
    w.set_data(_numbers_table())
    assert w.Warning.no_text.is_shown()
    w.start_training()  # must not crash, must not spawn a worker
    assert w.worker is None


def test_trainer_outputs_commit_on_done(qapp, capture, text_table):
    """Full pipeline in-thread (blocking)."""
    w = OWLMPretrainer()
    w.show()
    w.set_data(text_table)
    w.epochs = 2
    w.mlp_context_width = 8
    w.mlp_embedding_dim = 8
    w.mlp_hidden_units = 16
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


def test_worker_stop_flag_honoured(qapp, capture, text_table, monkeypatch):
    """The Stop button's effect, mid-run: 200 epochs requested, stop after 3."""
    w = OWLMPretrainer()
    w.show()
    w.set_data(text_table)
    w.epochs = 200
    w.mlp_context_width = 8
    w.mlp_embedding_dim = 8
    w.mlp_hidden_units = 16

    original = w._on_epoch
    calls = {"n": 0}

    def on_epoch(stats):  # the button handler runs outside the worker; the
        original(stats)  # hook is the only mid-run injection point here
        calls["n"] += 1
        if calls["n"] >= 3:
            w.stop_training()

    monkeypatch.setattr(w, "_on_epoch", on_epoch)
    model_sent = capture(w.Outputs.model)
    w.run_training_blocking()
    assert 3 <= len(w.epoch_stats) < 200  # ended early, not crashed
    assert w.model is not None and model_sent[-1] is w.model


def test_stop_button_sets_flag(qapp, text_table):
    w = OWLMPretrainer()
    w.show()
    w.set_data(text_table)
    w.worker = TrainingWorker([], None, None, {})
    w.stop_training()
    assert w.worker.request_stop is True


def test_failure_shows_the_last_traceback_line(qapp):
    w = OWLMPretrainer()
    w.show()
    w._on_failed("Traceback (most recent call last):\n  ...\nRuntimeError: kapow")
    assert w.Error.training_failed.is_shown()
    assert "kapow" in w.Error.training_failed.formatted
    assert not w.Information.training.is_shown()


def test_training_failure_surfaces_as_message(qapp, text_table, monkeypatch):
    w = OWLMPretrainer()
    w.show()
    w.set_data(text_table)

    def failing_train(*args, **kwargs):
        raise RuntimeError("kapow")

    monkeypatch.setattr(
        "orangecontrib.ai_suite.widgets.ow_lm_pretrainer.train_language_model", failing_train
    )
    w.epochs = 2
    w.run_training_blocking()
    assert w.Error.training_failed.is_shown()
    assert not w.Information.training.is_shown()


def test_only_the_selected_architectures_controls_are_visible(qapp):
    """Every architecture owns its hyperparameters, so the others' are hidden."""
    w = OWLMPretrainer()
    w.show()
    for index, expected in enumerate(["bigram", "mlp", "lstm"]):
        w.architecture_index = index
        w._show_architecture()
        visible = [name for name, box in w._arch_boxes.items() if box.isVisible()]
        assert visible == [expected]


def test_architecture_is_built_from_its_own_settings_only(qapp):
    w = OWLMPretrainer()
    w.mlp_context_width = 8
    w.mlp_embedding_dim, w.mlp_hidden_units = 8, 16
    w.lstm_context_width = 12
    w.lstm_embedding_dim, w.lstm_units, w.lstm_layers = 12, 48, 2

    w.architecture_index = 1
    assert w._architecture().get_params() == {
        "context_width": 8,
        "embedding_dim": 8,
        "hidden_units": 16,
    }
    w.architecture_index = 2
    assert w._architecture().get_params() == {
        "context_width": 12,
        "embedding_dim": 12,
        "units": 48,
        "layers": 2,
    }
    w.architecture_index = 0
    assert w._architecture().get_params() == {}  # a bigram has no hyperparameters
