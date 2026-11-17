"""LM Vocab Viewer: a read-only view of a tokenizer.

Input is a model (preferred: its embedded tokenizer is shown) or data with text
columns (then the character and word tokenizers are built side by side,
making the LM Pre-trainer's tokenizer choice visible). There are deliberately no
outputs: the viewer never feeds back into training or generation, so
vocabulary-mismatch bugs are impossible by construction.

The viewer shows the token-to-id table, corpus frequency counts (a Zipf's
law side lesson) and its longest tokens.
"""

from __future__ import annotations

from AnyQt.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Msg, OWWidget

from orangecontrib.ai_suite.adapters import corpus_text


def _fill_table(table_widget: QTableWidget, rows: list[tuple]) -> None:
    table_widget.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table_widget.setItem(r, c, QTableWidgetItem(str(value)))


class OWLMVocabViewer(OWWidget):
    """Inspect a tokenizer: tokens, ids and frequencies.

    Connect a model to see the vocabulary embedded in it, or connect text data
    to compare a character and a word tokenizer built from that
    data. The summary line reports vocabulary size, mean token length and the
    fraction of tokens seen only once; the tabs list the full vocabulary for
    each tokenizer.

    Read-only by design: there are no output signals.
    """

    name = "LM Vocab Viewer"
    description = "Inspect a model's tokenizer: tokens, ids, frequencies."
    icon = "icons/lm_vocab_viewer.svg"
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
        tabs.addTab(self.char_table, "Char")
        tabs.addTab(self.word_table, "Word")
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
                text = corpus_text(table)
                from orangecontrib.ai_suite.kernel.lm import tokenizer_from_text  # pure Python

                self._char_tokenizer = tokenizer_from_text(text, "char")
                self._word_tokenizer = tokenizer_from_text(text, "word")
            except ValueError as exc:
                self.Warning.text_not_supported(str(exc))
        self.refresh()

    # ---- rendering ---------------------------------------------------------

    def refresh(self):
        if self._model_tokenizer is not None:
            self._show_tokenizer(self._model_tokenizer, self.model_table)
            stats = self._model_tokenizer.stats()
            dropped = stats["dropped_occurrences"]
            self.summary_label.setText(
                f"{stats['tokenizer_name']} tokenizer, vocab {stats['vocab_size']} "
                f"(incl. specials), mean token length {stats['mean_token_length']:.2f}, "
                f"hapax fraction {stats['single_occurrence_fraction']:.0%}"
                + (f", {dropped} corpus tokens dropped" if dropped else "")
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
        """Rows in rank order: vocabulary order is meaningless, frequency is not."""
        specials = [
            (tokenizer.token(i), i, tokenizer.counts[i]) for i in range(tokenizer.n_special)
        ]
        ranked = [(tok, tokenizer.index(tok), count) for tok, count in tokenizer.ranked_counts()]
        _fill_table(table, specials + ranked)


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLMVocabViewer).run()
