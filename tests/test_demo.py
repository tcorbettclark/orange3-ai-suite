"""Demo corpus cleaner: Volume One extraction from a Gutenberg text."""

import pytest

from demo.fetch_gibbon import clean

RAW = """The Project Gutenberg eBook of something

*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

Table of contents

      Introduction.

      This is the first paragraph,
      wrapped across two lines.
      1 (return) [ a footnote that
      must disappear. ]
      The second paragraph follows.
      2 (return)
      [ a footnote whose bracket
      starts on the next line. ]
      The third paragraph continues.
      VOLUME TWO
      This must not appear.

*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

Project Gutenberg footer.
"""


def test_volume_one_extraction_and_note_removal():
    corpus = clean(RAW)
    assert "first paragraph, wrapped across two lines" in corpus
    assert "second paragraph follows" in corpus
    assert "third paragraph continues" in corpus
    assert "footnote" not in corpus
    assert "VOLUME TWO" not in corpus
    assert "Project Gutenberg" not in corpus


def test_non_gutenberg_text_is_rejected():
    with pytest.raises(ValueError, match="Gutenberg"):
        clean("just some text")
