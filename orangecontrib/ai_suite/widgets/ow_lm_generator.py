"""LM Generator: a table of generated text from any model that streams tokens.

Any object exposing ``stream(prompt, temperature, max_new_tokens, seed)`` is an
input: edu_ai_suite's Keras ``LanguageModel``, a ``LoRAModel``, or a saved artifact
loaded from disk. The text appears as the model writes it, Stop takes effect at the
next chunk, and the Generations output is a one-column Table.
"""

from __future__ import annotations

import traceback

from AnyQt.QtCore import QThread, pyqtSignal
from AnyQt.QtWidgets import QPlainTextEdit
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.ai_suite.adapters import generations_table


class _GenerationWorker(QThread):
    """Streams the model's text off the GUI thread: one signal per chunk."""

    chunk = pyqtSignal(int, str)  # generation index, text delta
    finished_ok = pyqtSignal(object)  # list[str]
    failed = pyqtSignal(str)

    def __init__(self, model, prompt, temperature, max_new_tokens, n_generations, seed):
        super().__init__()
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._n_generations = n_generations
        self._seed = seed
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            outcomes = []
            for index in range(self._n_generations):
                seed = None if self._seed < 0 else self._seed + index
                parts = []
                for piece in self._model.stream(
                    self._prompt, self._temperature, self._max_new_tokens, seed
                ):
                    if self._stop:
                        break
                    parts.append(piece)
                    self.chunk.emit(index, piece)
                outcomes.append("".join(parts))
                if self._stop:
                    break
            self.finished_ok.emit(outcomes)
        except Exception:
            self.failed.emit(traceback.format_exc())


class OWLMGenerator(OWWidget):
    """Generate text from a trained model that streams tokens.

    Input is a model object (Keras, MLX/LoRA, or compatible); the controls set the
    prompt, temperature, maximum new tokens, number of generations and seed, and the
    Generations output carries a one-column Table.

    For a LoRA model, "Use LoRA adapter" switches between the adapted model and its
    raw base model.
    """

    name = "LM Generator"
    description = "Generate text with a trained language model."
    icon = "icons/lm_generator.svg"
    priority = 20
    keywords = ["generate", "sampling", "temperature", "language model"]  # noqa: RUF012

    class Inputs:
        model = Input("Model", object, auto_summary=False)

    class Outputs:
        generations = Output("Generations", Table)

    class Warning(OWWidget.Warning):
        not_generative = Msg("The connected model does not expose stream().")

    class Error(OWWidget.Error):
        generation_failed = Msg("Generation failed: {}")

    prompt = Setting("the")
    temperature = Setting(0.8)
    max_new_tokens = Setting(48)
    n_generations = Setting(3)
    seed = Setting(0)
    use_adapter = Setting(True)

    def __init__(self):
        super().__init__()
        self.model = None

        box = gui.widgetBox(self.controlArea, "Prompt")
        gui.lineEdit(box, self, "prompt", label="Prompt text")
        gui.hSlider(
            box,
            self,
            "temperature",
            label="Temperature",
            minValue=0.0,
            maxValue=3.0,
            step=0.1,
            intOnly=False,
            labelFormat=" %.1f",
        )
        gui.spin(box, self, "max_new_tokens", 1, 512, label="Max new tokens")
        gui.spin(box, self, "n_generations", 1, 50, label="Number of generations")
        gui.spin(box, self, "seed", -1, 2**31 - 1, label="Seed (-1 = random)")
        self._adapter_check = gui.checkBox(
            self.controlArea,
            self,
            "use_adapter",
            "Use LoRA adapter (when available)",
        )
        self._adapter_check.setEnabled(False)
        buttons = gui.hBox(self.controlArea)
        self._generate_button = gui.button(buttons, self, "Generate", callback=self.generate)
        self._stop_button = gui.button(buttons, self, "Stop", callback=self.stop_generation)
        self._stop_button.setEnabled(False)
        gui.rubber(self.controlArea)

        box = gui.widgetBox(self.mainArea, "Generations")
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlaceholderText("Press Generate: the text appears here as it is written.")
        box.layout().addWidget(self.view)

        self.worker: _GenerationWorker | None = None
        self._texts: list[str] = []

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.Warning.not_generative(
            shown=model is not None and not callable(getattr(model, "stream", None))
        )
        supports_adapter = hasattr(model, "adapter_enabled")
        self._adapter_check.setEnabled(supports_adapter)
        if supports_adapter:
            self.use_adapter = bool(model.adapter_enabled)

    def generate(self):
        self._run(blocking=False)

    def run_generation_blocking(self):
        """Same-thread variant for tests/debugging."""
        self._run(blocking=True)

    def _run(self, blocking: bool):
        if self.worker is not None and self.worker.isRunning():
            return
        self.Outputs.generations.send(None)
        self.Error.generation_failed.clear()
        if self.model is None or not callable(getattr(self.model, "stream", None)):
            return
        if hasattr(self.model, "adapter_enabled"):
            self.model.adapter_enabled = self.use_adapter
        self.view.clear()
        self._texts = []
        worker = _GenerationWorker(
            self.model,
            self.prompt,
            self.temperature,
            self.max_new_tokens,
            self.n_generations,
            self.seed,
        )
        worker.chunk.connect(self._on_chunk)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self.worker = worker
        self._set_running(True)
        if blocking:
            worker.run()
        else:
            worker.start()

    def stop_generation(self):
        if self.worker is not None:
            self.worker.stop()

    def _set_running(self, running: bool) -> None:
        """Enable Generate or Stop, according to whether a run is in progress."""
        self._generate_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def _on_chunk(self, index: int, text: str) -> None:
        if index >= len(self._texts):
            self._texts.append(text)
        else:
            self._texts[index] += text
        self.view.setPlainText("\n\n".join(self._texts))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_done(self, outcomes) -> None:
        self._set_running(False)
        self._texts = list(outcomes)
        if self._texts:
            self.view.setPlainText("\n\n".join(self._texts))
        self.Outputs.generations.send(generations_table(list(outcomes)))

    def _on_failed(self, traceback_text: str) -> None:
        self._set_running(False)
        detail = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "?"
        self.Error.generation_failed(detail)


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLMGenerator).run()
