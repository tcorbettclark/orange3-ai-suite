"""LM Trainer: pre-train a tiny language model on text data (spec §4.1).

A thin shell over the kernel: options, a live loss/perplexity plot, and the
checkpoint table. All substance (tokenizer, windowing, Keras builders,
sampling) lives in nn-kernel; the widget runs training in a background
QThread and streams per-epoch stats into a live pyqtgraph plot — the
established QThread + callback pattern (orange3-imagenets precedent,
pattern only, its CC-BY-NC-SA code never touched).

The kernel is imported lazily: without the [lm] extra (Keras), the widget
shows a clear install hint instead of crashing (graceful degradation,
spec §3).
"""

from __future__ import annotations

import traceback

import numpy as np
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget, Output
from AnyQt.QtCore import QThread, pyqtSignal

from orangecontrib.nn_suite.adapters import checkpoint_table, table_texts

ARCHITECTURE_LABELS = ["Bigram", "MLP-over-window", "LSTM"]
ARCHITECTURE_VALUES = ["bigram", "mlp_window", "lstm"]
LEVEL_LABELS = ["Char level", "Word level"]
LEVEL_VALUES = ["char", "word"]


class TrainingWorker(QThread):
    """Runs kernel training off the GUI thread; stats stream via signals."""

    epoch_signal = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, texts, tokenizer, options, parent=None):
        super().__init__(parent)
        self.texts = texts
        self.tokenizer = tokenizer
        self.options = options
        self.request_stop = False

    def run(self):  # noqa: D102 — called off-thread
        try:
            from nn_kernel.lm import train_language_model  # lazy: Keras heavy

            model = train_language_model(
                self.texts,
                self.tokenizer,
                self.options["architecture"],
                context_size=self.options["context_size"],
                embedding_dim=self.options["embedding_dim"],
                hidden_units=self.options["hidden_units"],
                lstm_units=self.options["lstm_units"],
                lstm_layers=self.options["lstm_layers"],
                epochs=self.options["epochs"],
                batch_size=self.options["batch_size"],
                validation_fraction=self.options["validation_fraction"],
                learning_rate=self.options["learning_rate"],
                seed=self.options["seed"],
                on_epoch=self.epoch_signal.emit,  # thread-safe queued signal
                stop_requested=lambda: self.request_stop,
                sample_prompt=self.options["sample_prompt"],
                sample_temperature=self.options["sample_temperature"],
                sample_max_new_tokens=self.options["sample_max_new_tokens"],
                sample_seed=self.options["sample_seed"],
            )
            self.finished_ok.emit(model)
        except Exception:  # pragma: no cover - exact failure surfaces via msg
            self.failed.emit(traceback.format_exc())


class OWLMTrainer(OWWidget):
    name = "LM Trainer"
    description = "Pre-train a tiny language model (bigram / MLP / LSTM) on text data."
    icon = "icons/lm_trainer.svg"
    priority = 10
    keywords = ["pre-training", "language model", "keras", "perplexity"]  # noqa: RUF012

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)
        checkpoint = Output("Checkpoint", Table)

    class Warning(OWWidget.Warning):
        no_text = Msg("Connect data with string (text) columns, or a Corpus.")
        nothing_to_train = Msg("Press Train once data is connected.")

    class Error(OWWidget.Error):
        keras_missing = Msg(
            "Keras is not available. "
            "Install the [lm] extra: pip install orange3-nn-suite[lm]"
        )
        training_failed = Msg("Training failed:\n{}")

    class Information(OWWidget.Information):
        training = Msg("Training in progress; press Stop to end early.")

    # settings
    architecture_index = Setting(1)  # MLP-over-window (spec: flagship default)
    tokenizer_level_index = Setting(0)  # char (small vocab, the visible experiment)
    context_size = Setting(16)
    embedding_dim = Setting(32)
    hidden_units = Setting(128)
    lstm_units = Setting(256)
    lstm_layers_index = Setting(0)  # combo: 1 or 2 layers
    epochs = Setting(15)
    batch_size = Setting(256)
    learning_rate = Setting(1e-3)
    validation_fraction = Setting(0.1)
    max_vocab = Setting(0)  # 0 = unlimited
    store_samples = Setting(True)
    sample_prompt = Setting("the")
    sample_temperature = Setting(0.0)  # greedy: stable, comparable across epochs
    sample_max_new_tokens = Setting(24)

    def __init__(self):
        super().__init__()
        self.data = None
        self.texts: list[str] | None = None
        self.worker: TrainingWorker | None = None
        self.model = None
        self.epoch_stats: list[dict] = []

        arch_box = gui.widgetBox(self.controlArea, "Architecture")
        gui.comboBox(arch_box, self, "architecture_index",
                     items=ARCHITECTURE_LABELS, label="Architecture")
        gui.comboBox(arch_box, self, "tokenizer_level_index",
                     items=LEVEL_LABELS, label="Tokenizer")
        gui.spin(arch_box, self, "context_size", 1, 64, label="Context window")
        gui.spin(arch_box, self, "embedding_dim", 2, 512, label="Embedding dim")
        gui.spin(arch_box, self, "hidden_units", 8, 1024, label="Hidden units (MLP)")

        lstm_box = gui.widgetBox(self.controlArea, "LSTM")
        gui.spin(lstm_box, self, "lstm_units", 16, 1024, label="LSTM units")
        gui.comboBox(lstm_box, self, "lstm_layers_index", items=["1 layer", "2 layers"])

        train_box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(train_box, self, "epochs", 1, 500, label="Epochs")
        gui.spin(train_box, self, "batch_size", 1, 4096, label="Batch size")
        gui.doubleSpin(train_box, self, "learning_rate", 1e-5, 0.1, 1e-4,
                       label="Learning rate")
        gui.doubleSpin(train_box, self, "validation_fraction", 0.01, 0.5, 0.01,
                       label="Validation fraction")
        gui.spin(train_box, self, "max_vocab", 0, 200000, label="Max vocab (0 = all)")

        sample_box = gui.widgetBox(self.controlArea, "Checkpoint samples")
        gui.checkBox(sample_box, self, "store_samples", "Record a sample each epoch")
        gui.lineEdit(sample_box, self, "sample_prompt", label="Sample prompt")
        gui.doubleSpin(sample_box, self, "sample_temperature", 0.0, 3.0, 0.1,
                       label="Sample temperature")
        gui.spin(sample_box, self, "sample_max_new_tokens", 1, 128,
                 label="Sample length")

        gui.button(self.controlArea, self, "Train", callback=self.start_training)
        gui.button(self.controlArea, self, "Stop", callback=self.stop_training)
        self.status_label = gui.label(self.controlArea, self, "")

        # live plot: train and validation loss (perplexity lives in the
        # checkpoint table and status text; e^loss is the interpretable number)
        import pyqtgraph as pg

        self._plot = pg.PlotWidget(background="w")
        self._plot.addLegend(offset=(10, 10))
        self._plot.setLabel("bottom", "Epoch")
        self._plot.setLabel("left", "Loss")
        self._train_curve = self._plot.plot(pen=pg.mkPen("#1f77b4", width=2),
                                            name="train loss")
        self._val_curve = self._plot.plot(pen=pg.mkPen("#d62728", width=2),
                                          name="val loss (dashed)")
        self.mainArea.layout().addWidget(self._plot)
        self.resize(1000, 700)

    @Inputs.data
    def set_data(self, table):
        self.data = table
        self.texts = None
        self.Warning.no_text.clear()
        if table is not None:
            try:
                self.texts = table_texts(table)
            except ValueError as exc:
                self.Warning.no_text()
        self.Warning.nothing_to_train(shown=self.texts is None)

    # ---- lifecycle ---------------------------------------------------------

    def _collect_options(self) -> dict:
        return {
            "architecture": ARCHITECTURE_VALUES[self.architecture_index],
            "tokenizer_level": LEVEL_VALUES[self.tokenizer_level_index],
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "hidden_units": self.hidden_units,
            "lstm_units": self.lstm_units,
            "lstm_layers": self.lstm_layers_index + 1,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_fraction": self.validation_fraction,
            "max_vocab": self.max_vocab or None,
            "seed": 0,
            "sample_prompt": self.sample_prompt,
            "sample_temperature": self.sample_temperature if self.store_samples else None,
            "sample_max_new_tokens": self.sample_max_new_tokens,
        }

    def start_training(self):
        if self.texts is None:
            self.Warning.nothing_to_train()
            return
        self.Outputs.model.send(None)
        self.Outputs.checkpoint.send(None)
        self.model = None
        self.epoch_stats = []
        from nn_kernel.lm import Tokenizer  # pure Python; never imports Keras

        options = self._collect_options()
        tokenizer = Tokenizer.from_corpus(self.texts, options.pop("tokenizer_level"),
                                          max_vocab=options["max_vocab"])
        self.worker = TrainingWorker(self.texts, tokenizer, options)
        self.worker.epoch_signal.connect(self._on_epoch)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.Information.training()

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop = True

    def run_training_blocking(self):
        """Run the worker in the calling thread (used by tests / debugging)."""
        if self.texts is None:
            return
        self.Outputs.model.send(None)
        self.Outputs.checkpoint.send(None)
        self.model = None
        self.epoch_stats = []
        from nn_kernel.lm import Tokenizer

        options = self._collect_options()
        tokenizer = Tokenizer.from_corpus(self.texts, options.pop("tokenizer_level"),
                                          max_vocab=options["max_vocab"])
        self.worker = TrainingWorker(self.texts, tokenizer, options)
        self.worker.epoch_signal.connect(self._on_epoch)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.run()

    # ---- live feedback -------------------------------------------------------

    def _on_epoch(self, stats: dict):
        self.epoch_stats.append(stats)
        epochs = [s["epoch"] for s in self.epoch_stats]
        self._train_curve.setData(epochs, [s["loss"] for s in self.epoch_stats])
        finite_val = [(s["epoch"], s["val_loss"]) for s in self.epoch_stats
                      if np.isfinite(s["val_loss"])]
        if finite_val:
            self._val_curve.setData(*zip(*finite_val))
        self.status_label.setText(
            f"epoch {stats['epoch']}: loss {stats['loss']:.3f}, "
            f"val {stats['val_loss']:.3f}, perplexity {stats['perplexity']:.1f}"
        )

    def _on_done(self, model):
        self.Information.training.clear()
        self.model = model
        self.Outputs.model.send(model)
        self.Outputs.checkpoint.send(checkpoint_table(self.epoch_stats))
        hp = model.hyperparameters
        self.status_label.setText(
            f"Done: {hp['architecture']}, {hp['n_train_windows']} train windows "
            f"({hp['n_val_windows']} val), "
            f"final val perplexity {hp['final_val_loss']:.3f} -> "
            f"{np.exp(hp['final_val_loss']):.1f}"
        )

    def _on_failed(self, tb: str):
        self.Information.training.clear()
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        broken_import = "ModuleNotFoundError" in tb or (
            "ImportError" in tb and "halted" in tb
        )
        if broken_import and ("keras" in tb or "tensorflow" in tb):
            self.Error.keras_missing()
        else:
            self.Error.training_failed(last)

    def onDeleteWidget(self):
        if self.worker is not None:
            self.worker.request_stop = True
            self.worker.wait(3000)
        super().onDeleteWidget()