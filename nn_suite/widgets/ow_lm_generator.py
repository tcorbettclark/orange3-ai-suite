"""LM Generator: generate text with any kernel-convention model.

The input is any object exposing ``generate(prompt, temperature,
max_new_tokens, seed)``: the kernel's Keras ``LanguageModel``, a
``LoRAModel``, or a saved artifact loaded from disk. The output is a
one-column Table of generations.

Temperature is the most instructive knob in language-model demos: low values
repeat, high values dissolve into word salad.
"""

from __future__ import annotations

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from nn_suite.adapters import generations_table


class OWLMGenerator(OWWidget):
    """Generate text from a trained kernel-convention model.

    Input is a model object (Keras, MLX/LoRA, or compatible). The controls set
    the prompt, temperature, maximum new tokens, number of generations and
    seed. Generations are emitted as a one-column Table, ready for Corpus
    Viewer and the Orange3-Text analysis widgets.

    ``Load model...`` opens a saved artifact directory: a Keras LM
    (``meta.json``) or a LoRA adapter (``lora_meta.json``). The loaded model
    stays active until the input signal replaces it.
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
        not_generative = Msg("The connected model does not expose generate().")

    class Error(OWWidget.Error):
        load_failed = Msg("Could not load the model: {}")
        generation_failed = Msg("Generation failed: {}")

    class Information(OWWidget.Information):
        loaded = Msg("Loaded {} model: {}.")

    prompt = Setting("the")
    temperature = Setting(0.8)
    max_new_tokens = Setting(48)
    n_generations = Setting(3)
    seed = Setting(0)

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
        gui.button(self.controlArea, self, "Load model...", callback=self.load_model)
        gui.button(self.controlArea, self, "Generate", callback=self.generate)
        gui.rubber(self.controlArea)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.Error.load_failed.clear()
        self.Warning.not_generative(
            shown=model is not None and not callable(getattr(model, "generate", None))
        )

    def load_model(self):
        """Load a saved model artifact directory."""
        from pathlib import Path

        self.Error.load_failed.clear()

        from AnyQt.QtWidgets import QFileDialog

        src_dir = QFileDialog.getExistingDirectory(
            self, "Select a saved model/adapter artifact directory"
        )
        if not src_dir:
            return
        src = Path(src_dir)
        try:
            if (src / "lora_meta.json").exists():
                from nn_kernel.lora import LoRAModel

                model, kind = LoRAModel.from_artifact(src), "MLX LoRA"
            elif (src / "meta.json").exists():
                from nn_kernel.lm import LanguageModel

                model, kind = LanguageModel.load(src), "Keras LM"
            else:
                self.Warning.not_generative()
                return
        except Exception as exc:
            self.Error.load_failed(f"{type(exc).__name__}: {exc}")
            return
        self.set_model(model)
        self.Information.loaded(kind, src.name)

    def generate(self):
        self.Outputs.generations.send(None)
        self.Error.generation_failed.clear()
        if self.model is None or not callable(getattr(self.model, "generate", None)):
            return
        seed = self.seed
        try:
            outcomes = [
                self.model.generate(
                    self.prompt,
                    self.temperature,
                    self.max_new_tokens,
                    None if seed < 0 else seed + i,
                )
                for i in range(self.n_generations)
            ]
        except Exception as exc:
            self.Error.generation_failed(f"{type(exc).__name__}: {exc}")
            return
        self.Outputs.generations.send(generations_table(outcomes))


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLMGenerator).run()
