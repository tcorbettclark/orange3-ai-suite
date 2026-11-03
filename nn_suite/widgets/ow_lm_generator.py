"""LM Generator: generate text with any kernel-convention model (spec §4.2).

Input: any model object exposing the kernel convention
``generate(prompt, temperature, max_new_tokens, seed)`` — the Keras
``LanguageModel`` today, MLX/LoRA models of Stages 2-3 later (identical
interface, by design).

Output: the generations as a one-column text Table.
Temperature is the most instructive knob in LM demos: low = repetitive,
high = word salad.
"""

from __future__ import annotations

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget, Output

from nn_suite.adapters import generations_table


# ``object`` inputs auto_summarize=None keeps Orange from registering a
# summarizer for arbitrary model objects (a cosmetic warning otherwise).
class OWLMGenerator(OWWidget):
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

    class Information(OWWidget.Information):
        hint = Msg("Set a prompt, then press Generate.")

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
        gui.hSlider(box, self, "temperature", label="Temperature",
                    minValue=0.0, maxValue=3.0, step=0.1, intOnly=False,
                    labelFormat=" %.1f")
        gui.spin(box, self, "max_new_tokens", 1, 512, label="Max new tokens")
        gui.spin(box, self, "n_generations", 1, 50, label="Number of generations")
        gui.spin(box, self, "seed", -1, 2**31 - 1, label="Seed (-1 = random)")
        gui.button(self.controlArea, self, "Generate", callback=self.generate)
        gui.rubber(self.controlArea)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.Warning.not_generative(
            shown=model is not None and not callable(getattr(model, "generate", None))
        )

    def generate(self):
        self.Outputs.generations.send(None)
        if self.model is None or not callable(getattr(self.model, "generate", None)):
            return
        seed = self.seed
        outcomes = []
        for i in range(self.n_generations):
            use_seed = None if seed < 0 else seed + i
            # The kernel convention: generate(prompt, temperature,
            # max_new_tokens, seed) — positionally, so MLX models adhere too.
            outcomes.append(
                self.model.generate(
                    self.prompt, self.temperature, self.max_new_tokens, use_seed
                )
            )
        self.Outputs.generations.send(generations_table(outcomes))
        self.Information.hint(shown=False)