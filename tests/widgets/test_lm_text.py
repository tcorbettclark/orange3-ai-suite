"""LM Text widget: file/folder reading, row modes, Q/A pairs and preview."""

from AnyQt.QtWidgets import QPlainTextEdit

from nn_suite.adapters import table_texts
from nn_suite.widgets.ow_lm_text import OWLMText


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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
    assert [m.name for m in out.domain.metas] == ["text", "source"]
    assert table_texts(out) == ["First paragraph.", "Second paragraph."]
    assert out.metas[0, 1] == "book.txt"
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
    assert sorted(str(v) for v in out.metas[:, 1]) == ["a.txt", "sub/b.txt"]


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
    assert [m.name for m in out.domain.metas] == ["prompt", "completion", "source"]
    assert out.metas[0, 0] == "Q: Who is the narrator?\nA:"
    assert out.metas[0, 1] == "A woman in an ancestral hall."
    assert out.metas[1, 1] == "The yellow wallpaper."
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
