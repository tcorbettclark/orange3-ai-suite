"""LM Generator widget: prompt/temperature controls, generations as a Table."""

from Orange.data import Table

from orangecontrib.ai_suite.widgets.ow_lm_generator import OWLMGenerator


class StubModel:
    """Duck-typed kernel convention: stream(prompt, temperature, max, seed).

    Yields in word-sized pieces, which is all the widget may assume: the real
    streams chunk wherever the library does.
    """

    def __init__(self, pieces=("the", " cat", " sat"), on_chunk=None):
        self.calls = []
        self.pieces = pieces
        self.on_chunk = on_chunk

    def stream(self, prompt, temperature=1.0, max_new_tokens=64, seed=None):
        self.calls.append((prompt, temperature, max_new_tokens, seed))
        yield prompt
        for piece in self.pieces:
            yield piece
            if self.on_chunk is not None:
                self.on_chunk()


def test_generate_passes_the_kernel_convention_positionally(qapp, capture):
    w = OWLMGenerator()
    w.show()
    model = StubModel()
    w.set_model(model)
    sent = capture(w.Outputs.generations)
    w.prompt = "the"
    w.temperature = 0.5
    w.max_new_tokens = 12
    w.n_generations = 2
    w.seed = 7
    w.run_generation_blocking()
    # stream(prompt, temperature, max_new_tokens, seed) positionally, seeds
    # advancing per generation
    assert model.calls[0] == ("the", 0.5, 12, 7)
    assert model.calls[1] == ("the", 0.5, 12, 8)
    table = sent[-1]
    assert isinstance(table, Table) and len(table) == 2
    from orangecontrib.ai_suite.adapters import generations_strings

    assert generations_strings(table)[0] == "thethe cat sat"


def test_seed_minus_one_does_not_pass_a_seed(qapp, capture):
    w = OWLMGenerator()
    w.show()
    model = StubModel()
    w.set_model(model)
    w.seed = -1
    w.n_generations = 1
    w.run_generation_blocking()
    assert model.calls[0][3] is None


def test_no_model_or_bad_model_sends_none(qapp, capture):
    w = OWLMGenerator()
    w.show()
    sent = capture(w.Outputs.generations)
    w.generate()  # no model
    assert sent[-1] is None
    assert w.Warning.not_generative.is_shown() is False


def test_text_arrives_chunk_by_chunk_in_the_view(qapp, capture):
    # The point of streaming: the text is readable while the model is writing,
    # not only when it has finished.
    seen = []
    w = OWLMGenerator()
    w.show()
    model = StubModel(pieces=("abc", "def"))
    model.on_chunk = lambda: seen.append(w.view.toPlainText())
    w.set_model(model)
    w.prompt = ""
    w.n_generations = 1
    w.run_generation_blocking()
    assert "abc" in seen[-1]
    assert w.view.toPlainText() == "abcdef"
    assert w._generate_button.isEnabled() and not w._stop_button.isEnabled()


def test_stop_keeps_what_was_written_so_far(qapp, capture):
    w = OWLMGenerator()
    w.show()
    sent = capture(w.Outputs.generations)
    model = StubModel(pieces=("one", "two", "three"))
    model.on_chunk = w.stop_generation  # a student pressing Stop after one chunk
    w.set_model(model)
    w.prompt = ""
    w.n_generations = 1
    w.run_generation_blocking()

    from orangecontrib.ai_suite.adapters import generations_strings

    assert generations_strings(sent[-1]) == ["one"]
    assert not w._stop_button.isEnabled()


def test_a_model_without_stream_is_reported(qapp, capture):
    class OnlyGenerate:
        def generate(self, prompt, temperature=1.0, max_new_tokens=64, seed=None):
            return "text"

    w = OWLMGenerator()
    w.show()
    sent = capture(w.Outputs.generations)
    w.set_model(OnlyGenerate())
    assert w.Warning.not_generative.is_shown()
    w.generate()
    assert sent[-1] is None

    w.set_model(object())  # connected but not generative
    assert w.Warning.not_generative.is_shown()
    w.generate()
    assert sent[-1] is None


def test_output_type_is_table(qapp, capture, text_table):
    # the generations output is a plain Orange Table (readable in Corpus Viewer)
    w = OWLMGenerator()
    w.show()
    model = StubModel()
    w.set_model(model)
    sent = capture(w.Outputs.generations)
    w.prompt = "x"
    w.n_generations = 1
    w.run_generation_blocking()
    assert sent[-1].domain.metas[0].name == "generation"


class AdapterStub(StubModel):
    def __init__(self):
        super().__init__()
        self.adapter_enabled = True


def test_adapter_toggle_is_only_for_adapter_models(qapp, capture):
    w = OWLMGenerator()
    w.show()
    plain = StubModel()
    w.set_model(plain)
    assert not w._adapter_check.isEnabled()

    model = AdapterStub()
    w.set_model(model)
    assert w._adapter_check.isEnabled()
    assert w.use_adapter is True

    w.use_adapter = False
    w.n_generations = 1
    w.run_generation_blocking()
    assert model.adapter_enabled is False
