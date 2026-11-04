"""LoRA Trainer: fine-tune a real LLM with 4-bit QLoRA (Apple Silicon).

A thin shell over :mod:`nn_kernel.lora`. The widget adds the QThread and live
loss curve, the base-model selector, the two data formats and the output
artifact (adapter + base-model reference as one unit).

The data-format choice is a teaching beat:

- "Raw corpus" - continued pre-training, still self-supervised;
- "Prompt-response pairs" - supervised fine-tuning, with the prompt masked
  out of the loss.

MLX is required (macOS on Apple Silicon); other machines get the standard
graceful-degradation message.
"""

from __future__ import annotations

import traceback

from AnyQt.QtCore import QThread, pyqtSignal
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from nn_suite.adapters import table_pairs, table_texts

DATA_FORMATS = ("Raw corpus (continued pre-training)", "Prompt-response pairs (SFT)")
DEFAULT_BASE_MODEL = "britllm/britllm-3b-v0.1"


class LoraWorker(QThread):
    """Runs kernel LoRA training off the GUI thread; metrics stream via signals."""

    metric_signal = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, request, parent=None):
        super().__init__(parent)
        self.request = request  # kwargs for the kernel's train_lora
        self.request_stop_flag = False

    def run(self):
        try:
            from nn_kernel.lora import train_lora  # guarded import inside

            adapter = train_lora(
                on_metric=self.metric_signal.emit,
                stop_requested=lambda: self.request_stop_flag,
                **self.request,
            )
            self.finished_ok.emit(adapter)
        except Exception:
            self.failed.emit(traceback.format_exc())


class OWLoRATrainer(OWWidget):
    """Fine-tune a pretrained model with 4-bit QLoRA.

    Input is a Table of raw text (one document per row) or prompt-response
    pairs. Output is a ``LoRAModel``: the base model plus the trained adapter,
    ready for the LM Generator.

    The "Data format" choice is the lesson. A raw corpus trains with the same
    next-token loss as pre-training (continued pre-training). Prompt-response
    pairs are supervised fine-tuning: the completion tokens carry the loss and
    the prompt is masked out. The live curve shows loss per iteration; the
    adapter's trainable-parameter share is a small fraction of the base
    model's parameters.
    """

    name = "LoRA Trainer"
    description = "Fine-tune a real LLM with 4-bit QLoRA (Apple Silicon)."
    icon = "icons/lora_trainer.svg"
    priority = 15
    keywords = ["lora", "fine-tuning", "sft", "mlx", "qlora"]  # noqa: RUF012

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", object, auto_summary=False)

    class Warning(OWWidget.Warning):
        no_text = Msg("No usable training data: {}")
        nothing_to_train = Msg("Press Train once data is connected.")
        fp16_model = Msg(
            "The base model is not 4-bit quantised. Convert it first: "
            "uv run python demo/fetch_britllm.py"
        )

    class Error(OWWidget.Error):
        mlx_missing = Msg(
            "MLX not available: LoRA training needs macOS on Apple Silicon "
            "with the [lora] extra (pip install orange3-nn-suite[lora])."
        )
        training_failed = Msg("Training failed:\n{}")

    class Information(OWWidget.Information):
        training = Msg("Training in progress; press Stop to end early.")

    # settings
    base_model = Setting(DEFAULT_BASE_MODEL)
    data_format_index = Setting(0)  # raw corpus is the classroom default
    iters = Setting(300)
    learning_rate = Setting(1e-4)
    num_layers = Setting(16)
    batch_size = Setting(4)
    max_seq_length = Setting(512)
    seed = Setting(42)

    def __init__(self):
        super().__init__()
        self.data = None
        self.worker: LoraWorker | None = None
        self.model = None
        self._losses: list[tuple[int, float, str]] = []  # (iter, loss, kind)
        self._trainable_note = ""

        model_box = gui.widgetBox(self.controlArea, "Model")
        gui.lineEdit(
            model_box,
            self,
            "base_model",
            label="Base model (path or HF id)",
            tooltip="Path to a local MLX model directory or a Hugging "
            "Face repo id. QLoRA requires a 4-bit quantised "
            "model; demo/fetch_britllm.py produces one.",
        )

        data_box = gui.widgetBox(self.controlArea, "Data format (the lesson)")
        gui.comboBox(data_box, self, "data_format_index", items=list(DATA_FORMATS))

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

    # ---- lifecycle ---------------------------------------------------------

    def _collect_request(self, tmp_dir) -> dict:
        from nn_kernel.lora import LoraData, LoraOptions

        if self.data_format_index == 1:
            data_obj = LoraData.from_pairs(table_pairs(self.data))
        else:
            data_obj = LoraData.from_texts(table_texts(self.data))
        options = LoraOptions(
            iters=self.iters,
            learning_rate=self.learning_rate,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            seed=self.seed,
        )
        return dict(
            base_model=self.base_model or DEFAULT_BASE_MODEL,
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
        self.Warning.fp16_model.clear()
        self.Error.mlx_missing.clear()
        self.Error.training_failed.clear()
        if self.data is None:
            self.Warning.nothing_to_train()
            return None
        if not self._mlx_available():
            self.Error.mlx_missing()
            return None
        base_model = self.base_model or DEFAULT_BASE_MODEL
        if not self._model_is_quantised(base_model):
            self.Warning.fp16_model()
            return None  # only 4-bit QLoRA is offered

        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lora_"))
        try:
            return self._collect_request(tmp_dir)
        except ValueError as exc:
            self.Warning.no_text(str(exc))
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

    @staticmethod
    def _model_is_quantised(base_model: str) -> bool:
        """Local directories: inspect config.json's quantization settings.

        A Hugging Face id may be quantised on the Hub, but that cannot be known
        without downloading the model card, so it is treated as unquantised.
        The 4-bit-only rule then points the classroom at the locally converted
        model.
        """
        import json
        from pathlib import Path

        candidate = Path(base_model)
        if not candidate.is_dir():
            return False  # HF id or missing path - not verifiably quantised
        config = candidate / "config.json"
        if not config.exists():
            return False
        return bool(json.loads(config.read_text()).get("quantization"))

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
        from nn_kernel.lora import LoRAModel

        self.model = LoRAModel(
            adapter.base_model,
            adapter.adapter_path,
            adapter.hyperparameters,
            adapter.name,
        )
        self.Outputs.model.send(self.model)
        detail = f" ({self._trainable_note})" if self._trainable_note else ""
        self.status_label.setText(
            f"Done: adapter trained ({adapter.name or 'adapter'}){detail}; "
            "connect the Model output to LM Generator for the before/after "
            "comparison."
        )

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


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLoRATrainer).run()
