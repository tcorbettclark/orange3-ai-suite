"""Vocab Inspector: a read-only window into a tokenizer (spec §4.4).

Inputs: a model (preferred — its embedded tokenizer is shown) or data with
text columns (then char-level and word-level tokenizers are built to run
side by side, the Trainer's tokenizer-dropdown hypothesis made visible).

There are deliberately NO outputs: the inspector never feeds back into
training or generation, which makes vocabulary-mismatch bugs impossible by
construction. The inspector shows the tokenizer's token->id table, corpus
frequency counts (Zipf's law), and its longest tokens.
"""

from __future__ import annotations

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Msg, OWWidget
from AnyQt.QtWidgets import QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTabWidget

from nn_suite.adapters import table_texts


def _fill_table(table_widget: QTableWidget, rows: list[tuple]) -> None:
    table_widget.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table_widget.setItem(r, c, QTableWidgetItem(str(value)))


class OWVocabInspector(OWWidget):
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
        text_not_supported = Msg("Data carries no string (text) columns.")
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
        self.Warning.no_tokenizer(
            shown=model is not None and self._model_tokenizer is None
        )
        self.Warning.model_preferred(shown=model is not None and self._data is not None)
        self.refresh()

    @Inputs.data
    def set_data(self, table):
        self._data = table
        self._char_tokenizer = self._word_tokenizer = None
        if table is not None:
            try:
                texts = table_texts(table)
                from nn_kernel.lm import Tokenizer  # kernel import; pure Python

                self._char_tokenizer = Tokenizer.from_corpus(texts, "char")
                self._word_tokenizer = Tokenizer.from_corpus(texts, "word")
            except ValueError as exc:
                self.Warning.text_not_supported(exc.args[0] if exc.args else str(exc))
        self.refresh()

    # ---- rendering ---------------------------------------------------------

    def refresh(self):
        if self._model_tokenizer is not None:
            self._show_tokenizer(self._model_tokenizer, self.model_table)
            from nn_kernel.lm import vocab_stats

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
            self.summary_label.setText("No input.")
            self._tabs.setCurrentIndex(0)

    @staticmethod
    def _show_tokenizer(tokenizer, table: QTableWidget) -> None:
        _fill_table(
            table,
            [
                (tokenizer.token(i), i, tokenizer.counts[i])
                for i in range(tokenizer.vocab_size)
            ],
        )


if __name__ == "__main__":  # pragma: no cover — dev preview (REPO-CONTEXT §4)
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWVocabInspector).run()
