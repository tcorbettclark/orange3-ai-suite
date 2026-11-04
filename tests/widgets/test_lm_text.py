"""LM Text widget: file/folder reading, row modes, Q/A pairs and preview."""

from AnyQt.QtWidgets import QPlainTextEdit

from nn_suite.adapters import table_texts
from nn_suite.widgets.ow_lm_text import OWLMText


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_new_instance_starts_empty(qapp):
    w = OWLMText()
    w.show()
    assert w._rows == []
    assert w._table.rowCount() == 0
    assert w._text.toPlainText() == ""


def test_document_file_rows_and_preview(qapp, capture, tmp_path):
    path = _write(tmp_path / "book.txt", "First paragraph.\n\nSecond paragraph.")
    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.source_type = 0
    w._path_edit.setText(str(path))
    w._reload()

    out = sent[-1]
    assert out is not None
    assert len(out) == 2
    assert [m.name for m in out.domain.metas] == ["text"]
    assert table_texts(out) == ["First paragraph.", "Second paragraph."]
    assert w._table.rowCount() == 2
    assert w._text.toPlainText() == "First paragraph."
    assert w._text.lineWrapMode() == QPlainTextEdit.WidgetWidth


def test_folder_is_recursive_and_keeps_source(qapp, capture, tmp_path):
    root = tmp_path / "corpus"
    _write(root / "a.txt", "Alpha.")
    _write(root / "sub" / "b.txt", "Beta.")
    _write(root / "ignore.md", "not text")

    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.source_type = 1
    w._path_edit.setText(str(root))
    w._reload()

    out = sent[-1]
    assert out is not None
    assert len(out) == 2
    assert table_texts(out) == ["Alpha.", "Beta."]
    # The source is still visible in the preview, but not in the Data output.
    assert sorted(w._table.item(r, 0).text() for r in range(2)) == ["a.txt", "sub/b.txt"]


def test_qa_mode_extracts_pairs_and_warns_on_malformed(qapp, capture, tmp_path):
    path = _write(
        tmp_path / "pairs.txt",
        "Q: Who is the narrator?\nA: A woman in an ancestral hall.\n\n"
        "Q: This one has no answer\n"
        "Q: What is she obsessed with?\nA: The yellow wallpaper.\n",
    )
    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.table_mode = 1
    w.source_type = 0
    w._path_edit.setText(str(path))
    w._reload()

    out = sent[-1]
    assert out is not None
    assert len(out) == 2
    assert [m.name for m in out.domain.metas] == ["prompt", "completion"]
    assert out.metas[0, 0] == "Q: Who is the narrator?\nA:"
    assert out.metas[0, 1] == "A woman in an ancestral hall."
    assert out.metas[1, 1] == "The yellow wallpaper."
    assert w._table.item(0, 0).text() == "pairs.txt"
    assert w.Warning.malformed_pairs.is_shown()


def test_reload_reads_changed_file(qapp, capture, tmp_path):
    path = _write(tmp_path / "live.txt", "Only one.")
    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.source_type = 0
    w._path_edit.setText(str(path))
    w._reload()
    assert len(sent[-1]) == 1

    _write(path, "One.\n\nTwo.\n\nThree.")
    w._reload()
    assert len(sent[-1]) == 3


def test_preview_is_capped_with_explanation(qapp, capture, tmp_path):
    path = _write(
        tmp_path / "many.txt",
        "\n".join(f"line {i}" for i in range(205)),
    )
    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.source_type = 0
    w.split_mode = 2  # lines
    w._path_edit.setText(str(path))
    w._reload()

    assert len(sent[-1]) == 205
    assert w._table.rowCount() == 200
    assert "first 200 of 205" in w._preview_label.text()
    assert "Data output contains all 205" in w._preview_label.text()


def test_gutenberg_cleaning_is_applied(qapp, capture, tmp_path):
    path = _write(
        tmp_path / "gutenberg.txt",
        "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Body text.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter\n",
    )
    w = OWLMText()
    w.show()
    sent = capture(w.Outputs.data)
    w.source_type = 0
    w.strip_gutenberg = True
    w._path_edit.setText(str(path))
    w._reload()
    assert table_texts(sent[-1]) == ["Body text."]
