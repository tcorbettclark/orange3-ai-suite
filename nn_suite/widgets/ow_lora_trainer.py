"""LoRA Trainer: fine-tune a real LLM with 4-bit QLoRA (spec §4.3).

The Stage-3 widget. A thin shell over nn_kernel.lora, which wraps the
documented `python -m mlx_lm lora` command in a stopped-streamable
subprocess — the widget adds the QThread + live-loss pattern, the model
selector, the two data formats (raw corpus = continued pre-training;
prompt–response pairs = true SFT with prompt masking), and the output
artifact (adapter + base model as one unit).

Two teaching beats are exposed as data format choice in the GUI:
- "Raw corpus (continued pre-training)" — self-supervised;
- "Prompt–response pairs (SFT)" — supervised, prompt-masked loss.

Apple-only: MLX is required (macOS/Apple Silicon); other machines get the
standard graceful-degradation message.
"""

from __future__ import annotations

import traceback

import numpy as np
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget, Output
from AnyQt.QtCore import QThread, pyqtSignal

from nn_suite.adapters import table_pairs, table_texts

DATA_FORMATS = ("Raw corpus (continued pre-training)", "Prompt–response pairs (SFT)")
DEFAULT_BASE_MODEL = "britllm/britllm-3b-v0.1"


class LoraWorker(QThread):
    metric_signal = pyqtSignal(dict)
    line_signal = pyqtSignal(str)
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
                on_line=None,
                stop_requested=lambda: self.request_stop_flag,
                **self.request,
            )
            self.finished_ok.emit(adapter)
        except Exception:  # noqa: BLE001 — surfaced as a widget message
            self.failed.emit(traceback.format_exc())


class OWLoRATrainer(OWWidget):
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
        no_text = Msg("Connect data with text columns (or prompt/completion pairs).")
        nothing_to_train = Msg("Press Train once data is connected.")

    class Error(OWWidget.Error):
        mlx_missing = Msg(
            "MLX not available: LoRA training needs macOS on Apple Silicon "
            "with the [lora] extra (pip install orange3-nn-suite[lora])."
        )
        training_failed = Msg("Training failed:\n{}")

    class Information(OWWidget.Information):
        training = Msg("Training in progress; press Stop to end early.")

    # settings
    auto_apply = Setting(True)
    base_model = Setting(DEFAULT_BASE_MODEL)
    data_format_index = Setting(0)  # raw corpus (the classroom default beat)
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

        model_box = gui.widgetBox(self.controlArea, "Model")
        gui.lineEdit(model_box, self, "base_model",
                     label="Base model (path or HF id)",
                     tooltip="Path to a local MLX model dir (demo/fetch_britllm.py "
                             "converts britllm) or a Hugging Face repo id")

        data_box = gui.widgetBox(self.controlArea, "Data format (the lesson)")
        gui.comboBox(data_box, self, "data_format_index",
                     items=list(DATA_FORMATS))

        train_box = gui.widgetBox(self.controlArea, "Training")
        gui.spin(train_box, self, "iters", 1, 5000, label="Iterations")
        gui.doubleSpin(train_box, self, "learning_rate", 1e-6, 0.01, 1e-5,
                       label="Learning rate")
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
        self._val_curve = self._plot.plot(pen="#d62728", width=2, name="val loss",
                                          symbol="o", symbolSize=6, symbolPen="w")
        self.mainArea.layout().addWidget(self._plot)
        self.resize(1000, 700)

    @Inputs.data
    def set_data(self, table):
        self.data = table
        self.Warning.nothing_to_train(shown=table is None)
        self._request = None  # collected at Train time

    def _collect_request(self, tmp_dir) -> dict:
        from nn_kernel.lora import LoraData, LoraOptions

        if self.data_format_index == 1:
            pairs = table_pairs(self.data)
            data_obj = LoraData.from_pairs(pairs)
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

    def start_training(self):
        if self.data is None:
            self.Warning.nothing_to_train()
            return
        if not self._mlx_available():
            self.Error.mlx_missing()
            return
        self.Outputs.model.send(None)
        self.model = None
        self._losses = []
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])

        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lora_"))
        request = self._collect_request(tmp_dir)

        self.worker = LoraWorker(request)
        self.worker.metric_signal.connect(self._on_metric)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.Information.training()

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop_flag = True

    def run_training_blocking(self):
        """Same-thread variant for tests/debugging."""
        if self.data is None or not self._mlx_available():
            return
        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lora_"))
        self.Outputs.model.send(None)
        self.model = None
        self._losses = []
        request = self._collect_request(tmp_dir)
        self.worker = LoraWorker(request)
        self.worker.metric_signal.connect(self._on_metric)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.run()

    def _on_metric(self, metric: dict):
        self._losses.append((metric["iter"], metric["loss"], metric["kind"]))
        iters = [i for i, _, _ in self._losses]
        train = [(i, l) for i, l, k in self._losses if k == "train"]
        val = [(i, l) for i, l, k in self._losses if k == "val"]
        if train:
            self._train_curve.setData(*zip(*train))
        if val:
            self._val_curve.setData(*zip(*val))
        self.status_label.setText(
            f"iter {metric['iter']}: loss {metric['loss']:.3f}"
            + (f", {metric.get('tok_s')} tok/s" if metric.get("tok_s") else "")
        )

    def _on_done(self, adapter):
        self.Information.training.clear()
        from nn_kernel.lora import LoRAModel

        self.model = LoRAModel(adapter.base_model, adapter.adapter_path)
        self.Outputs.model.send(self.model)
        self.status_label.setText(
            f"Done: adapter saved ({adapter.name or 'adapter'}); "
            "connect the Model output to LM Generator for the before/after "
            "comparison (toggle the adapter)."
        )

    def _on_failed(self, tb: str):
        self.Information.training.clear()
        clean = tb.strip()
        last = clean.splitlines()[-1] if clean else "unknown error"
        if ("No module named" in last and ("mlx" in last or "mlx_lm" in last)) or \
                "Apple Silicon" in clean:
            self.Error.mlx_missing()
        else:
            self.Error.training_failed(last)

    def onDeleteWidget(self):
        if self.worker is not None:
            self.worker.request_stop_flag = True
            self.worker.wait(5000)
        super().onDeleteWidget()