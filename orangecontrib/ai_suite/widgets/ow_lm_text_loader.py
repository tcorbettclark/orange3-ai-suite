"""LM Text Loader: read plain-text files into document rows or Q/A pairs.

A source widget for the language-model teaching suite. It reads UTF-8 ``.txt``
files, applies opt-in cleaning, and emits a plain Orange Table. In document
mode each row is a whole file, paragraph, line or sentence. In Q/A mode each
row is one prompt/completion pair from fixed ``Q:``/``A:`` blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
)
from Orange.data import Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Msg, Output, OWWidget

from orangecontrib.ai_suite.kernel.text import (
    CleanOptions,
    clean_text,
    parse_qa,
    reflow_qa_pair,
    split_text,
)

MODES = ("Whole file", "Paragraphs", "Lines", "Sentences")
SPLIT_KEYS = ("whole", "paragraphs", "lines", "sentences")
PREVIEW_ROWS = 200


class OWLMTextLoader(OWWidget):
    """Read plain-text files into a Table for the language-model widgets.

    Source is a single ``.txt`` file or a recursively scanned folder. Document
    mode emits one row per file, paragraph, line or sentence, with a ``text``
    meta. Q/A mode emits ``prompt`` and ``completion`` metas from fixed blocks:
    a new pair starts at ``Q:``, the prompt includes the following ``A:`` line,
    and the completion runs to the next ``Q:``. The source file is shown in the
    preview but is not included in the Data output.
    """

    name = "LM Text Loader"
    description = "Read plain-text files into documents or Q/A pairs."
    icon = "icons/lm_text_loader.svg"
    priority = 20
    keywords = ["text", "corpus", "documents", "language model", "q/a"]  # noqa: RUF012

    class Outputs:
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        source_missing = Msg("Choose a .txt file or a folder of .txt files.")
        source_not_found = Msg("Source path does not exist: {}")
        no_txt_files = Msg("No .txt files found in this folder.")
        unreadable_files = Msg("{} file(s) could not be read as UTF-8.")
        no_rows = Msg("No text rows produced. Check the table type and cleaning options.")
        malformed_pairs = Msg("{} Q/A block(s) skipped (missing Q:, A:, or empty field).")

    source_type = Setting(0)  # 0 = file, 1 = folder
    source_path = Setting("")
    table_mode = Setting(0)  # 0 = documents, 1 = Q/A pairs
    split_mode = Setting(1)  # default: paragraphs
    strip_gutenberg = Setting(False)
    reflow = Setting(False)
    remove_repeated_lines = Setting(False)
    remove_page_numbers = Setting(False)

    def __init__(self):
        super().__init__()
        self._raw_files: list[tuple[str, str]] = []
        self._rows: list[tuple] = []

        source_box = gui.widgetBox(self.controlArea, "Source")
        gui.comboBox(
            source_box,
            self,
            "source_type",
            items=("File", "Folder"),
            label="Type",
            callback=self._source_type_changed,
        )
        path_row = QHBoxLayout()
        source_box.layout().addLayout(path_row)
        self._path_edit = QLineEdit(self.source_path)
        self._path_edit.setPlaceholderText("Choose a .txt file or folder")
        self._path_edit.returnPressed.connect(self._reload)
        path_row.addWidget(self._path_edit, 1)
        self._browse_button = QPushButton("Browse...")
        self._browse_button.clicked.connect(self._browse)
        path_row.addWidget(self._browse_button)
        self._reload_button = QPushButton("Reload")
        self._reload_button.clicked.connect(self._reload)
        path_row.addWidget(self._reload_button)

        gui.radioButtons(
            self.controlArea,
            self,
            "table_mode",
            btnLabels=("Text documents", "Q/A pairs"),
            box="Table type",
            callback=self._sync_mode,
        )

        self._doc_box = gui.widgetBox(self.controlArea, "Document rows")
        gui.comboBox(
            self._doc_box,
            self,
            "split_mode",
            items=MODES,
            label="Split into",
            callback=self._rebuild,
        )

        self._qa_box = gui.widgetBox(self.controlArea, "Q/A pairs")
        example = QLabel(
            "Q: Who is the narrator?\n"
            "A: A woman spending the summer in an ancestral hall.\n\n"
            "Q: What does she become obsessed with?\n"
            "A: The yellow wallpaper in her room."
        )
        example.setWordWrap(True)
        example.setToolTip(
            "A new pair starts at a line beginning Q:. The prompt includes the "
            "A: line; the completion runs to the next Q: line."
        )
        self._qa_box.layout().addWidget(example)

        clean_box = gui.widgetBox(self.controlArea, "Cleaning (opt-in)")
        gui.checkBox(
            clean_box,
            self,
            "strip_gutenberg",
            "Strip Project Gutenberg header/footer",
            callback=self._rebuild,
        )
        gui.checkBox(
            clean_box,
            self,
            "reflow",
            "Reflow wrapped lines",
            callback=self._rebuild,
        )
        gui.checkBox(
            clean_box,
            self,
            "remove_repeated_lines",
            "Remove repeated short lines",
            callback=self._rebuild,
        )
        gui.checkBox(
            clean_box,
            self,
            "remove_page_numbers",
            "Remove page-number lines",
            callback=self._rebuild,
        )

        self._stats = QLabel("No text loaded.")
        self._stats.setWordWrap(True)
        self.controlArea.layout().addWidget(self._stats)
        self.controlArea.layout().addStretch(1)

        self._preview_label = QLabel("No rows.")
        self._preview_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self._preview_label)

        splitter = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(splitter)
        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.currentCellChanged.connect(self._show_current_row)
        splitter.addWidget(self._table)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._text.setPlaceholderText("Select a row to read its full text.")
        splitter.addWidget(self._text)
        splitter.setSizes([260, 360])

        self._sync_mode()
        self.resize(850, 700)
        if self.source_path:
            self._stats.setText("No text loaded. Press Reload to read the selected path.")

    def _source_type_changed(self):
        self._reload()

    def _browse(self):
        if self.source_type == 0:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose a text file", self._path_edit.text(), "Text files (*.txt)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Choose a folder of text files", self._path_edit.text()
            )
        if path:
            self._path_edit.setText(path)
            self._reload()

    def _reload(self):
        self.source_path = self._path_edit.text().strip()
        self.Warning.source_missing.clear()
        self.Warning.source_not_found.clear()
        self.Warning.no_txt_files.clear()
        self.Warning.unreadable_files.clear()
        self._raw_files = []
        if not self.source_path:
            self.Warning.source_missing()
            self._rebuild()
            return
        root = Path(self.source_path)
        if not root.exists():
            self.Warning.source_not_found(str(root))
            self._rebuild()
            return

        if self.source_type == 0:
            if root.suffix.lower() != ".txt":
                self.Warning.no_txt_files()
                self._rebuild()
                return
            paths = [root]
        else:
            paths = sorted(
                (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"),
                key=lambda p: str(p.relative_to(root)).lower(),
            )
            if not paths:
                self.Warning.no_txt_files()
                self._rebuild()
                return

        unreadable = 0
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                unreadable += 1
                continue
            source = str(path.relative_to(root)) if self.source_type == 1 else path.name
            self._raw_files.append((source, text))
        self.Warning.unreadable_files(unreadable, shown=bool(unreadable))
        self._rebuild()

    def _sync_mode(self):
        documents = self.table_mode == 0
        self._doc_box.setVisible(documents)
        self._qa_box.setVisible(not documents)
        self._rebuild()

    def _clean_options(self, reflow: bool) -> CleanOptions:
        return CleanOptions(
            strip_gutenberg=self.strip_gutenberg,
            reflow=reflow,
            remove_repeated_lines=self.remove_repeated_lines,
            remove_page_numbers=self.remove_page_numbers,
        )

    def _rebuild(self):
        self.Warning.no_rows.clear()
        self.Warning.malformed_pairs.clear()
        if not self._raw_files:
            self._rows = []
            self._populate_preview()
            self._text.clear()
            self._stats.setText("No text loaded.")
            self.Outputs.data.send(None)
            return

        removed_lines = 0
        if self.table_mode == 0:
            mode = SPLIT_KEYS[self.split_mode]
            rows = []
            for source, raw in self._raw_files:
                cleaned = clean_text(raw, self._clean_options(reflow=self.reflow))
                removed_lines += cleaned.removed_lines
                rows.extend((source, text) for text in split_text(cleaned.text, mode))
            self._rows = rows
            self._send_documents(rows)
            total_chars = sum(len(text) for _, text in rows)
            self._stats.setText(
                f"{len(self._raw_files)} file(s) -> {len(rows)} row(s), "
                f"{total_chars:,} characters; {removed_lines} line(s) removed."
            )
        else:
            rows = []
            skipped = 0
            for source, raw in self._raw_files:
                cleaned = clean_text(raw, self._clean_options(reflow=False))
                removed_lines += cleaned.removed_lines
                parsed = parse_qa(cleaned.text)
                skipped += parsed.skipped
                pairs = (
                    (reflow_qa_pair(pair) for pair in parsed.pairs) if self.reflow else parsed.pairs
                )
                for pair in pairs:
                    rows.append((source, pair.prompt, pair.completion))
            self._rows = rows
            self.Warning.malformed_pairs(skipped, shown=bool(skipped))
            self._send_pairs(rows)
            total_chars = sum(len(p) + len(c) for _, p, c in rows)
            self._stats.setText(
                f"{len(self._raw_files)} file(s) -> {len(rows)} pair(s), "
                f"{total_chars:,} characters; {skipped} skipped; "
                f"{removed_lines} line(s) removed."
            )

        if not self._rows:
            self.Warning.no_rows()
        self._populate_preview()

    def _send_documents(self, rows):
        if not rows:
            self.Outputs.data.send(None)
            return
        domain = Domain([], metas=[StringVariable("text")])
        metas = np.asarray([[text] for _source, text in rows], dtype=object)
        self.Outputs.data.send(Table.from_numpy(domain, np.empty((len(rows), 0)), None, metas))

    def _send_pairs(self, rows):
        if not rows:
            self.Outputs.data.send(None)
            return
        domain = Domain(
            [],
            metas=[StringVariable("prompt"), StringVariable("completion")],
        )
        metas = np.asarray(
            [[prompt, completion] for _source, prompt, completion in rows],
            dtype=object,
        )
        self.Outputs.data.send(Table.from_numpy(domain, np.empty((len(rows), 0)), None, metas))

    def _populate_preview(self):
        self._table.clear()
        total = len(self._rows)
        if not total:
            self._preview_label.setText("No rows.")
        elif total <= PREVIEW_ROWS:
            self._preview_label.setText(f"Showing all {total:,} rows.")
        else:
            self._preview_label.setText(
                f"Showing the first {PREVIEW_ROWS:,} of {total:,} rows. "
                f"The Data output contains all {total:,} rows."
            )
        if self.table_mode == 0:
            headers = ("Source", "Text")
        else:
            headers = ("Source", "Prompt", "Completion")
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        preview = self._rows[:PREVIEW_ROWS]
        self._table.setRowCount(len(preview))
        for r, row in enumerate(preview):
            if self.table_mode == 0:
                source, text = row
                cells = (source, _short(text))
            else:
                source, prompt, completion = row
                cells = (source, _short(prompt), _short(completion))
            for c, value in enumerate(cells):
                self._table.setItem(r, c, QTableWidgetItem(value))
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, len(headers)):
            header.setSectionResizeMode(c, QHeaderView.Stretch)
        if preview:
            self._table.setCurrentCell(0, 0)
            self._show_current_row(0, 0, -1, -1)
        else:
            self._text.clear()

    def _show_current_row(self, row, _column=0, _prev_row=-1, _prev_column=-1):
        if row < 0 or row >= len(self._rows):
            self._text.clear()
            return
        item = self._rows[row]
        if self.table_mode == 0:
            self._text.setPlainText(item[1])
        else:
            _, prompt, completion = item
            self._text.setPlainText(f"Prompt:\n{prompt}\n\nCompletion:\n{completion}")


def _short(text: str, limit: int = 180) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLMTextLoader).run()
