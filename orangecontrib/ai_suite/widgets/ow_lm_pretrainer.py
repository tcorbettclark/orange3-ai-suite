"""LM Pre-trainer: a trained language model and a table of per-epoch stats.

Architecture and tokenizer options, hyperparameters and a live loss plot; training
runs in a background QThread that streams each epoch's stats into pyqtgraph.
"""

from __future__ import annotations

import traceback

import numpy as np
from AnyQt.QtCore import QThread, pyqtSignal
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.ai_suite.adapters import checkpoint_table, corpus_text
from orangecontrib.ai_suite.kernel.lm import (
    LanguageModel,
    TrainingOptions,
    architectures,
    tokenizer_from_text,
    tokenizers,
)

# Names and labels come from the kernel's own discovery, so neither dropdown can
# drift from the classes it offers; only the label formatting is decided here.
ARCHITECTURES = architectures()
ARCHITECTURE_VALUES = [cls.architecture_name() for cls, _ in ARCHITECTURES]
ARCHITECTURE_LABELS = [cls.__name__ for cls, _ in ARCHITECTURES]
TOKENIZERS = tokenizers()
TOKENIZER_VALUES = [cls.tokenizer_name() for cls, _ in TOKENIZERS]
TOKENIZER_LABELS = [cls.__name__.removesuffix("Tokenizer") for cls, _ in TOKENIZERS]


class TrainingWorker(QThread):
    """Runs kernel training off the GUI thread; stats stream via signals."""

    epoch_signal = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, text, model, options, sample=None, parent=None):
        super().__init__(parent)
        self.text = text
        self.model = model
        self.options = options
        self.sample = sample  # (model) -> str, or None: the checkpoint column
        self.request_stop = False

    def _on_epoch(self, stats):
        """Stream one epoch's numbers, plus a sample if the widget asked for one."""
        if self.sample is not None:
            stats["sample"] = self.sample(self.model)
        self.epoch_signal.emit(stats)  # thread-safe queued signal

    def run(self):
        try:
            model = self.model.fit(
                self.text,
                self.options,
                on_epoch=self._on_epoch,
                stop_requested=lambda: self.request_stop,
            )
            self.finished_ok.emit(model)
        except Exception:  # pragma: no cover - exact failure surfaces via msg
            self.failed.emit(traceback.format_exc())


class OWLMPretrainer(OWWidget):
    """Pre-train a small language model and stream its learning curve.

    Input is a Table with text columns (or an Orange3-Text Corpus). The Model output
    is the trained model object with its tokenizer embedded; the Checkpoint output is
    one row per epoch with loss, perplexity and, when samples are recorded, a sample
    of generated text.

    The architecture dropdown selects the bigram, MLP or LSTM model, and the tokenizer
    dropdown the character or word tokenizer.
    """

    name = "LM Pre-trainer"
    description = "Pre-train a tiny language model (bigram / MLP / LSTM) from scratch on text data."
    icon = "icons/lm_pretrainer.svg"
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
        training_failed = Msg("Training failed:\n{}")

    class Information(OWWidget.Information):
        training = Msg("Training in progress; press Stop to end early.")

    # settings
    architecture_index = Setting(1)  # MLP is the default
    tokenizer_index = Setting(0)  # char: small vocab, long sequences
    # One setting per (architecture, hyperparameter), named after the estimator
    # parameter it feeds, so a model can never be handed a control that belongs
    # to another. Defaults agree on embedding_dim and context_width: a fair
    # comparison holds them fixed unless the user changes one.
    mlp_context_width = Setting(16)
    mlp_embedding_dim = Setting(32)
    mlp_hidden_units = Setting(128)
    lstm_context_width = Setting(16)
    lstm_embedding_dim = Setting(32)
    lstm_units = Setting(256)
    lstm_layers = Setting(1)
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
        self.text: str | None = None
        self.worker: TrainingWorker | None = None
        self.model = None
        self.epoch_stats: list[dict] = []

        arch_box = gui.widgetBox(self.controlArea, "Architecture")
        gui.comboBox(
            arch_box,
            self,
            "architecture_index",
            items=ARCHITECTURE_LABELS,
            label="Architecture",
            callback=self._show_architecture,
        )
        gui.comboBox(arch_box, self, "tokenizer_index", items=TOKENIZER_LABELS, label="Tokenizer")

        # Only the selected architecture's controls are shown: the others would
        # be inert, since each estimator declares just its own parameters.
        self._arch_boxes = {}
        bigram_box = gui.widgetBox(self.controlArea, "Bigram")
        gui.label(
            bigram_box,
            self,
            "No hyperparameters: the vocabulary-size count table, reading one token of history.",
        )
        self._arch_boxes["bigram"] = bigram_box

        mlp_box = gui.widgetBox(self.controlArea, "MLP")
        gui.spin(mlp_box, self, "mlp_context_width", 1, 64, label="Context width")
        gui.spin(mlp_box, self, "mlp_embedding_dim", 2, 512, label="Embedding dim")
        gui.spin(mlp_box, self, "mlp_hidden_units", 8, 1024, label="Hidden units")
        self._arch_boxes["mlp"] = mlp_box

        lstm_box = gui.widgetBox(self.controlArea, "LSTM")
        gui.spin(lstm_box, self, "lstm_context_width", 1, 64, label="Context width")
        gui.spin(lstm_box, self, "lstm_embedding_dim", 2, 512, label="Embedding dim")
        gui.spin(lstm_box, self, "lstm_units", 16, 1024, label="LSTM units")
        gui.spin(lstm_box, self, "lstm_layers", 1, 2, label="LSTM layers")
        self._arch_boxes["lstm"] = lstm_box
        self._show_architecture()

        train_box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(train_box, self, "epochs", 1, 500, label="Epochs")
        gui.spin(train_box, self, "batch_size", 1, 4096, label="Batch size")
        gui.doubleSpin(train_box, self, "learning_rate", 1e-5, 0.1, 1e-4, label="Learning rate")
        gui.doubleSpin(
            train_box, self, "validation_fraction", 0.01, 0.5, 0.01, label="Validation fraction"
        )
        gui.spin(train_box, self, "max_vocab", 0, 200000, label="Max vocab (0 = all)")

        sample_box = gui.widgetBox(self.controlArea, "Checkpoint samples")
        gui.checkBox(sample_box, self, "store_samples", "Record a sample each epoch")
        gui.lineEdit(sample_box, self, "sample_prompt", label="Sample prompt")
        gui.doubleSpin(
            sample_box, self, "sample_temperature", 0.0, 3.0, 0.1, label="Sample temperature"
        )
        gui.spin(sample_box, self, "sample_max_new_tokens", 1, 128, label="Sample length")

        gui.button(self.controlArea, self, "Train", callback=self.start_training)
        gui.button(self.controlArea, self, "Stop", callback=self.stop_training)
        self.status_label = gui.label(self.controlArea, self, "")

        # live plot: train and validation loss (perplexity is in the checkpoint
        # table and the status text; e^loss is the interpretable number)
        import pyqtgraph as pg

        self._plot = pg.PlotWidget(background="w")
        self._plot.addLegend(offset=(10, 10))
        self._plot.setLabel("bottom", "Epoch")
        self._plot.setLabel("left", "Loss")
        self._train_curve = self._plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="train loss")
        self._val_curve = self._plot.plot(pen=pg.mkPen("#d62728", width=2), name="val loss")
        self.mainArea.layout().addWidget(self._plot)
        self.resize(1000, 700)

    @Inputs.data
    def set_data(self, table):
        self.data = table
        self.text = None
        self.Warning.no_text.clear()
        if table is not None:
            try:
                self.text = corpus_text(table)
            except ValueError:
                self.Warning.no_text()
        self.Warning.nothing_to_train(shown=self.text is None)

    # ---- lifecycle ---------------------------------------------------------

    def _show_architecture(self):
        """Show only the selected architecture's controls."""
        selected = ARCHITECTURE_VALUES[self.architecture_index]
        for name, box in self._arch_boxes.items():
            box.setVisible(name == selected)

    def _architecture(self):
        """The selected architecture, configured from the settings it declares.

        Each setting is named after the estimator parameter it feeds, as
        ``f"{name}_{param}"``.
        """
        cls = ARCHITECTURES[self.architecture_index][0]
        name = cls.architecture_name()
        values = {param: getattr(self, f"{name}_{param}") for param in cls().get_params()}
        return cls(**values)

    def _training_options(self) -> TrainingOptions:
        """The run's settings, as the kernel's record of a run."""
        return TrainingOptions(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            validation_fraction=self.validation_fraction,
            seed=0,
        )

    def _epoch_sample(self):
        """One generation from the model as it stands, or None when ``store_samples`` is off."""
        if not self.store_samples:
            return None
        return lambda model: model.generate(
            self.sample_prompt,
            self.sample_temperature,
            self.sample_max_new_tokens,
            seed=0,
        )

    def _prepare_worker(self) -> TrainingWorker | None:
        self.Warning.nothing_to_train.clear()
        self.Error.training_failed.clear()
        if self.text is None:
            return None
        self.Outputs.model.send(None)
        self.Outputs.checkpoint.send(None)
        self.model = None
        self.epoch_stats = []
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])
        self.status_label.setText("")

        text_tokenizer = tokenizer_from_text(self.text, TOKENIZER_VALUES[self.tokenizer_index])
        if self.max_vocab:
            # real tokens only; the special tokens are always kept
            text_tokenizer = text_tokenizer.keep_top(self.max_vocab)
        model = LanguageModel(self._architecture(), text_tokenizer)
        worker = TrainingWorker(self.text, model, self._training_options(), self._epoch_sample())
        worker.epoch_signal.connect(self._on_epoch)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        return worker

    def start_training(self):
        if self.worker is not None and self.worker.isRunning():
            return
        worker = self._prepare_worker()
        if worker is None:
            self.Warning.nothing_to_train()
            return
        self.worker = worker
        worker.start()
        self.Information.training()

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop = True

    def run_training_blocking(self):
        """Run the worker in the calling thread (used by tests / debugging)."""
        worker = self._prepare_worker()
        if worker is None:
            return
        self.worker = worker
        worker.run()

    # ---- live feedback -------------------------------------------------------

    def _on_epoch(self, stats: dict):
        self.epoch_stats.append(stats)
        epochs = [s["epoch"] for s in self.epoch_stats]
        self._train_curve.setData(epochs, [s["loss"] for s in self.epoch_stats])
        finite_val = [
            (s["epoch"], s["val_loss"]) for s in self.epoch_stats if np.isfinite(s["val_loss"])
        ]
        if finite_val:
            self._val_curve.setData(*zip(*finite_val, strict=True))
        self.status_label.setText(
            f"epoch {stats['epoch']}: loss {stats['loss']:.3f}, "
            f"val {stats['val_loss']:.3f}, perplexity {stats['perplexity']:.1f}"
        )

    def _on_done(self, model):
        self.Information.training.clear()
        self.model = model
        self.Outputs.model.send(model)
        self.Outputs.checkpoint.send(checkpoint_table(self.epoch_stats))
        results = model.results
        self.status_label.setText(
            f"Done: {type(model.architecture).__name__}, {results.n_train_pairs} train pairs "
            f"({results.n_val_pairs} val), final val loss "
            f"{results.final_val_loss:.3f} (perplexity "
            f"{results.final_val_perplexity:.1f})"
        )

    def _on_failed(self, tb: str):
        self.Information.training.clear()
        self.Error.training_failed.clear()
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        self.Error.training_failed(last)

    def onDeleteWidget(self):
        if self.worker is not None:
            self.worker.request_stop = True
            self.worker.wait(3000)
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLMPretrainer).run()
