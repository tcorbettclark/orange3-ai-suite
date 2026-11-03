"""Vocab Inspector: a read-only window into a tokenizer.

Input is a model (preferred: its embedded tokenizer is shown) or data with text
columns (then character-level and word-level tokenizers are built side by side,
making the Trainer's tokenizer choice visible). There are deliberately no
outputs: the inspector never feeds back into training or generation, so
vocabulary-mismatch bugs are impossible by construction.

The inspector shows the token-to-id table, corpus frequency counts (a Zipf's
law side lesson) and its longest tokens.
"""

from __future__ import annotations

from AnyQt.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Msg, OWWidget

from nn_suite.adapters import table_texts


def _fill_table(table_widget: QTableWidget, rows: list[tuple]) -> None:
    table_widget.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table_widget.setItem(r, c, QTableWidgetItem(str(value)))


class OWVocabInspector(OWWidget):
    """Inspect a tokenizer: tokens, ids and frequencies.

    Connect a model to see the vocabulary embedded in it, or connect text data
    to compare a character-level and a word-level tokenizer built from that
    data. The summary line reports vocabulary size, mean token length and the
    fraction of tokens seen only once; the tabs list the full vocabulary for
    each tokenizer.

    Read-only by design: there are no output signals.
    """

    name = "Vocab Inspector"
    description = "Inspect a model's tokenizer: tokens, ids, frequencies."
    icon = "icons/vocab_inspector.svg"
    priority = 40
    keywords = ["vocabulary", "tokenizer", "tokens", "language model"]  # noqa: RUF012

    class Inputs:
        model = Input("Model", object, auto_summary=False)
        data = Input("Data", Table)

    class Warning(OWWidget.Warning):
        no_tokenizer = Msg("The connected model does not expose a tokenizer.")
        text_not_supported = Msg("{}")
        model_preferred = Msg("Model connected; the data input is ignored.")

    def __init__(self):
        super().__init__()
        self._data = None
        self._model_tokenizer = None
        self._char_tokenizer = None
        self._word_tokenizer = None

        box = gui.widgetBox(self.controlArea, "Summary")
        self.summary_label = QLabel("No input.")
        box.layout().addWidget(self.summary_label)

        tabs = QTabWidget(self)
        self.model_table = self._make_table(("token", "id", "count"))
        self.char_table = self._make_table(("token", "id", "count"))
        self.word_table = self._make_table(("token", "id", "count"))
        tabs.addTab(self.model_table, "Model tokenizer")
        tabs.addTab(self.char_table, "Char level")
        tabs.addTab(self.word_table, "Word level")
        self.mainArea.layout().addWidget(tabs)
        self._tabs = tabs
        self.resize(620, 700)

    @staticmethod
    def _make_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setColumnWidth(0, 260)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table

    @Inputs.model
    def set_model(self, model):
        self._model_tokenizer = getattr(model, "tokenizer", None)
        self.Warning.no_tokenizer(shown=model is not None and self._model_tokenizer is None)
        self.Warning.model_preferred(
            shown=self._model_tokenizer is not None and self._data is not None
        )
        self.refresh()

    @Inputs.data
    def set_data(self, table):
        self._data = table
        self._char_tokenizer = self._word_tokenizer = None
        self.Warning.text_not_supported.clear()
        self.Warning.model_preferred(shown=self._model_tokenizer is not None and table is not None)
        if table is not None:
            try:
                texts = table_texts(table)
                from nn_kernel.lm import Tokenizer  # kernel import; pure Python

                self._char_tokenizer = Tokenizer.from_corpus(texts, "char")
                self._word_tokenizer = Tokenizer.from_corpus(texts, "word")
            except ValueError as exc:
                self.Warning.text_not_supported(str(exc))
        self.refresh()

    # ---- rendering ---------------------------------------------------------

    def refresh(self):
        if self._model_tokenizer is not None:
            from nn_kernel.lm import vocab_stats

            self._show_tokenizer(self._model_tokenizer, self.model_table)
            stats = vocab_stats(self._model_tokenizer)
            self.summary_label.setText(
                f"{stats['level']}-level, vocab {stats['vocab_size']} "
                f"(incl. specials), mean token length {stats['mean_token_length']:.2f}, "
                f"hapax fraction {stats['single_occurrence_fraction']:.0%}"
            )
            self._tabs.setCurrentWidget(self.model_table)
        elif self._char_tokenizer is not None:
            self._show_tokenizer(self._char_tokenizer, self.char_table)
            self._show_tokenizer(self._word_tokenizer, self.word_table)
            self.summary_label.setText(
                f"Char vocab {self._char_tokenizer.vocab_size} "
                f"vs. word vocab {self._word_tokenizer.vocab_size} "
                "(from the connected data)"
            )
            self._tabs.setCurrentWidget(self.char_table)
        else:
            for table in (self.model_table, self.char_table, self.word_table):
                _fill_table(table, [])
            self.summary_label.setText("No input.")
            self._tabs.setCurrentIndex(0)

    @staticmethod
    def _show_tokenizer(tokenizer, table: QTableWidget) -> None:
        _fill_table(
            table,
            [(tokenizer.token(i), i, tokenizer.counts[i]) for i in range(tokenizer.vocab_size)],
        )


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWVocabInspector).run()
