"""Vocab Inspector widget: read-only, model-or-data input, NO outputs."""

from nn_suite.widgets.ow_vocab_inspector import OWVocabInspector


class NoTokenizerModel:
    pass


def _make_widget(qapp):
    w = OWVocabInspector()
    w.show()
    return w


def test_data_input_builds_both_tokenizer_views(qapp, text_table, texts):
    w = _make_widget(qapp)
    w.set_data(text_table)
    assert w._char_tokenizer is not None and w._word_tokenizer is not None
    assert "Char vocab" in w.summary_label.text()
    # char side sees every distinct token (the vocabulary), not every character
    assert w.char_table.rowCount() == w._char_tokenizer.vocab_size
    assert "the" in [w.word_table.item(r, 0).text() for r in range(w.word_table.rowCount())]


def test_model_input_preferred_over_data(qapp, text_table):
    from nn_kernel.lm import Tokenizer

    model = type("M", (), {})()
    model.tokenizer = Tokenizer.from_corpus(["hello world"], "word")
    w = _make_widget(qapp)
    w.set_data(text_table)
    w.set_model(model)
    assert "vocab" in w.summary_label.text()
    assert w.model_table.rowCount() == model.tokenizer.vocab_size
    assert w.Warning.model_preferred.is_shown()


def test_model_without_tokenizer_warns(qapp):
    w = _make_widget(qapp)
    w.set_model(NoTokenizerModel())
    assert w.Warning.no_tokenizer.is_shown()
    assert "No input." in w.summary_label.text()


def test_inspector_declares_no_outputs(qapp):
    # the inspector never enters the pipeline: no output signals
    from orangewidget.widget import Output

    outputs = getattr(OWVocabInspector, "Outputs", None)
    declared = []
    if outputs is not None:
        declared = [v for v in outputs.__dict__.values() if isinstance(v, Output)]
    assert declared == []


def test_textless_data_warns(qapp):
    import numpy as np
    from Orange.data import ContinuousVariable, Domain, Table

    dom = Domain([ContinuousVariable("x")])
    table = Table.from_numpy(dom, np.arange(4.0).reshape(-1, 1))
    w = _make_widget(qapp)
    w.set_data(table)
    assert w.Warning.text_not_supported.is_shown()
