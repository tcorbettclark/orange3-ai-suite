"""Shared widget machinery for the LoRA Pre-trainer and LoRA SFT widgets.

A thin shell over :mod:`orangecontrib.ai_suite.kernel.lora`. The base class owns the QThread, live
loss curve, base-model selector, hyperparameters and Model output. Subclasses
supply only the data contract and the explanatory data box.

MLX is required (macOS on Apple Silicon); other machines get the standard
graceful-degradation message.
"""

from __future__ import annotations

import traceback

from AnyQt.QtCore import QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.ai_suite.kernel.lora import (
    LoRAOptions,
    LoRAModel,
    TrainingStopped,
    train_lora,
)

_OPTION_DEFAULTS = LoRAOptions()


class LoraWorker(QThread):
    """Runs kernel LoRA training off the GUI thread; metrics stream via signals."""

    metric_signal = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)
    stopped = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, request, parent=None):
        super().__init__(parent)
        self.request = request  # kwargs for the kernel's train_lora
        self.request_stop_flag = False

    def run(self):
        try:
            adapter = train_lora(
                on_metric=self.metric_signal.emit,
                stop_requested=lambda: self.request_stop_flag,
                **self.request,
            )
            self.finished_ok.emit(adapter)
        except TrainingStopped as exc:
            self.stopped.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class LoRATrainerBase(OWWidget, openclass=True):
    """Shared LoRA training UI for the two LoRA widgets.

    Subclasses supply ``data_box_title``, ``data_help`` and ``_make_lora_data``.
    The base model arrives on the Base model input, so a connected BritLLM Models
    widget passes its local path explicitly; the line edit also accepts a path or
    an HF id. The output is a ``LoRAModel``: the base model plus the trained
    adapter, ready for the LM Generator. The live curve shows loss per iteration.
    """

    name = None  # not a registered widget; subclasses provide the UI

    data_box_title = "Data"
    data_help = ""

    class Inputs:
        data = Input("Data", Table)
        base_model = Input("Base model", object, auto_summary=False)

    class Outputs:
        model = Output("Model", object, auto_summary=False)

    class Warning(OWWidget.Warning):
        no_text = Msg("No usable training data: {}")
        nothing_to_train = Msg("Press Train once data is connected.")
        no_model = Msg(
            "No base model: connect BritLLM Models or paste an MLX model "
            "directory or Hugging Face id."
        )
        bad_settings = Msg("Invalid training settings: {}")

    class Error(OWWidget.Error):
        mlx_missing = Msg(
            "MLX not available: LoRA training needs macOS on Apple Silicon "
            "with the [lora] extra (pip install orange3-ai-suite[lora])."
        )
        training_failed = Msg("Training failed:\n{}")

    class Information(OWWidget.Information):
        training = Msg("Training in progress; press Stop to end early.")
        stopped = Msg("Training stopped before mlx-lm wrote an adapter; no model was produced.")

    # settings
    base_model = Setting("")
    iters = Setting(_OPTION_DEFAULTS.iters)
    learning_rate = Setting(_OPTION_DEFAULTS.learning_rate)
    num_layers = Setting(_OPTION_DEFAULTS.num_layers)
    batch_size = Setting(_OPTION_DEFAULTS.batch_size)
    max_seq_length = Setting(_OPTION_DEFAULTS.max_seq_length)
    seed = Setting(_OPTION_DEFAULTS.seed)

    def __init__(self):
        super().__init__()
        self.data = None
        self.worker: LoraWorker | None = None
        self.model = None
        self._losses: list[tuple[int, float, str]] = []  # (iter, loss, kind)
        self._trainable_note = ""

        model_box = gui.widgetBox(self.controlArea, "Model")
        self.base_model_edit = gui.lineEdit(
            model_box,
            self,
            "base_model",
            label="Base model (path or HF id)",
            tooltip="Path to a local MLX model directory or a Hugging Face "
            "repo id. Start with the 4-bit BritLLM model from BritLLM Models; "
            "other MLX models (8-bit or unquantised) work if the machine has "
            "the memory.",
        )
        model_help = QLabel(
            "Start with the 4-bit BritLLM model from BritLLM Models. "
            "Other MLX models work if the machine has the memory."
        )
        model_help.setWordWrap(True)
        model_box.layout().addWidget(model_help)

        data_box = gui.widgetBox(self.controlArea, self.data_box_title)
        if self.data_help:
            help_label = QLabel(self.data_help)
            help_label.setWordWrap(True)
            data_box.layout().addWidget(help_label)

        train_box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(train_box, self, "iters", 1, 5000, label="Iterations")
        gui.doubleSpin(train_box, self, "learning_rate", 1e-6, 0.01, 1e-5, label="Learning rate")
        gui.spin(train_box, self, "num_layers", 2, 32, label="Fine-tuned layers")
        gui.spin(train_box, self, "batch_size", 1, 32, label="Batch size")
        gui.spin(train_box, self, "max_seq_length", 64, 4096, label="Max seq length")
        gui.spin(train_box, self, "seed", 0, 2**31 - 1, label="Seed")

        gui.button(self.controlArea, self, "Train", callback=self.start_training)
        gui.button(self.controlArea, self, "Stop", callback=self.stop_training)
        self.status_label = gui.label(self.controlArea, self, "")

        import pyqtgraph as pg

        self._plot = pg.PlotWidget(background="w")
        self._plot.addLegend(offset=(10, 10))
        self._plot.setLabel("bottom", "Iteration")
        self._plot.setLabel("left", "Loss")
        self._train_curve = self._plot.plot(pen="#1f77b4", width=2, name="train loss")
        self._val_curve = self._plot.plot(
            pen="#d62728", width=2, name="val loss", symbol="o", symbolSize=6, symbolPen="w"
        )
        self.mainArea.layout().addWidget(self._plot)
        self.resize(1000, 700)

    @Inputs.data
    def set_data(self, table):
        self.data = table
        self.Warning.no_text.clear()
        self.Warning.nothing_to_train(shown=table is None)

    @Inputs.base_model
    def set_base_model(self, value):
        path = "" if value is None else str(value)
        self.base_model = path
        if getattr(self, "base_model_edit", None) is not None:
            self.base_model_edit.setText(path)
        self.Warning.no_model.clear()

    # ---- lifecycle ---------------------------------------------------------

    def _make_lora_data(self):
        """Build the kernel's ``LoRAData`` from the connected Table."""
        raise NotImplementedError

    def _collect_request(self, tmp_dir, data_obj) -> dict:
        options = LoRAOptions(
            iters=self.iters,
            learning_rate=self.learning_rate,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            seed=self.seed,
        )
        return dict(
            base_model=self.base_model.strip(),
            data=data_obj,
            adapter_dir=tmp_dir / "adapter",
            data_dir=tmp_dir / "staged-data",
            options=options,
        )

    @staticmethod
    def _mlx_available() -> bool:
        import importlib.util

        return (
            importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )

    def _build_request(self):
        """Validate state and stage data; None (with a message) when unusable."""
        self.Warning.nothing_to_train.clear()
        self.Warning.no_text.clear()
        self.Warning.no_model.clear()
        self.Warning.bad_settings.clear()
        self.Error.mlx_missing.clear()
        self.Error.training_failed.clear()
        self.Information.stopped.clear()
        if self.data is None:
            self.Warning.nothing_to_train()
            return None
        if not self._mlx_available():
            self.Error.mlx_missing()
            return None
        if not self.base_model.strip():
            self.Warning.no_model()
            return None

        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lora_"))
        try:
            data_obj = self._make_lora_data()
        except ValueError as exc:
            self.Warning.no_text(str(exc))
            return None
        try:
            return self._collect_request(tmp_dir, data_obj)
        except ValueError as exc:
            self.Warning.bad_settings(str(exc))
            return None

    def _run(self, blocking: bool):
        request = self._build_request()
        if request is None:
            return
        self.Outputs.model.send(None)
        self.model = None
        self._losses = []
        self._trainable_note = ""
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])

        worker = LoraWorker(request)
        worker.metric_signal.connect(self._on_metric)
        worker.finished_ok.connect(self._on_done)
        worker.stopped.connect(self._on_stopped)
        worker.failed.connect(self._on_failed)
        self.worker = worker
        if blocking:
            worker.run()
        else:
            worker.start()
            self.Information.training()

    def start_training(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._run(blocking=False)

    def run_training_blocking(self):
        """Same-thread variant for tests/debugging."""
        self._run(blocking=True)

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop_flag = True

    # ---- live feedback -------------------------------------------------------

    def _on_metric(self, metric: dict):
        if metric.get("kind") == "params":
            self._trainable_note = f"trainable {metric['pct']}% ({metric['detail']})"
            self.status_label.setText(self._trainable_note)
            return
        self._losses.append((metric["iter"], metric["loss"], metric["kind"]))
        train = [(i, loss) for i, loss, kind in self._losses if kind == "train"]
        val = [(i, loss) for i, loss, kind in self._losses if kind == "val"]
        if train:
            self._train_curve.setData(*zip(*train, strict=True))
        if val:
            self._val_curve.setData(*zip(*val, strict=True))
        self.status_label.setText(
            f"iter {metric['iter']}: loss {metric['loss']:.3f}"
            + (f", {metric.get('tok_s')} tok/s" if metric.get("tok_s") else "")
            + (f"; {self._trainable_note}" if self._trainable_note else "")
        )

    def _on_done(self, adapter):
        self.Information.training.clear()
        self.model = LoRAModel(
            adapter.base_model,
            adapter.adapter_path,
            adapter.training,
            adapter.name,
        )
        self.Outputs.model.send(self.model)
        detail = f" ({self._trainable_note})" if self._trainable_note else ""
        training = adapter.training
        stopped = (
            f" Stopped at iteration {training['iters_completed']} of "
            f"{training['options']['iters']}."
            if training.get("stopped")
            else ""
        )
        self.status_label.setText(
            f"Done: adapter trained ({adapter.name or 'adapter'}){detail}.{stopped} "
            "Connect the Model output to LM Generator for the before/after "
            "comparison."
        )

    def _on_stopped(self, message: str):
        self.Information.training.clear()
        self.Information.stopped()
        self.status_label.setText("Stopped; no adapter was written.")

    def _on_failed(self, tb: str):
        self.Information.training.clear()
        self.Error.mlx_missing.clear()
        self.Error.training_failed.clear()
        clean = tb.strip()
        last = clean.splitlines()[-1] if clean else "unknown error"
        if (
            "No module named" in last and ("mlx" in last or "mlx_lm" in last)
        ) or "Apple Silicon" in clean:
            self.Error.mlx_missing()
        else:
            self.Error.training_failed(last)

    def onDeleteWidget(self):
        if self.worker is not None:
            self.worker.request_stop_flag = True
            self.worker.wait(5000)
        super().onDeleteWidget()
