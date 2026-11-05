"""Plain-text cleaning, splitting and Q/A parsing."""

from orangecontrib.ai_suite.kernel.text import (
    CleanOptions,
    QAPair,
    clean_text,
    parse_qa,
    reflow_qa_pair,
    split_text,
)


def test_gutenberg_markers_are_stripped():
    raw = (
        "Project Gutenberg header\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
        "The body begins here.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
        "Footer text\n"
    )
    result = clean_text(raw, CleanOptions(strip_gutenberg=True))
    assert result.text == "The body begins here."
    assert result.gutenberg_stripped


def test_cleaning_removes_page_numbers_and_repeated_lines():
    raw = (
        "THE HEADER\n12\nfirst line\nTHE HEADER\nPage 13\nsecond line\nTHE HEADER\n14\nthird line\n"
    )
    result = clean_text(
        raw,
        CleanOptions(remove_page_numbers=True, remove_repeated_lines=True),
    )
    assert "HEADER" not in result.text
    assert "12" not in result.text
    assert "Page 13" not in result.text
    assert "first line" in result.text
    assert result.removed_lines == 6


def test_reflow_joins_lines_and_repairs_hyphens():
    raw = "A wrapped para-\ngraph continues here.\n\nAnother paragraph."
    result = clean_text(raw, CleanOptions(reflow=True))
    assert result.text == "A wrapped paragraph continues here.\n\nAnother paragraph."


def test_split_modes():
    text = "First sentence. Second sentence.\n\nA new paragraph."
    assert split_text(text, "whole") == (text,)
    assert split_text(text, "paragraphs") == (
        "First sentence. Second sentence.",
        "A new paragraph.",
    )
    assert split_text("one\ntwo\n", "lines") == ("one", "two")
    assert split_text(text, "sentences") == (
        "First sentence.",
        "Second sentence.",
        "A new paragraph.",
    )


def test_parse_qa_includes_marker_in_prompt():
    raw = (
        "Q: Who is the narrator?\n"
        "A: A woman spending the summer in an ancestral hall.\n\n"
        "Q: What does she become obsessed with?\n"
        "A: The yellow wallpaper in her room.\n"
    )
    result = parse_qa(raw)
    assert result.skipped == 0
    assert len(result.pairs) == 2
    assert result.pairs[0].prompt == "Q: Who is the narrator?\nA:"
    assert result.pairs[0].completion == ("A woman spending the summer in an ancestral hall.")
    assert result.pairs[1].completion == "The yellow wallpaper in her room."


def test_parse_qa_skips_malformed_blocks():
    raw = "Q: no answer here\nQ: ok?\nA: yes\nQ: empty\nA:\n"
    result = parse_qa(raw)
    assert len(result.pairs) == 1
    assert result.pairs[0].completion == "yes"
    assert result.skipped == 2


def test_parse_qa_allows_multiline_answer_and_blank_lines():
    raw = "Q: question\nA: first line\n\nsecond paragraph\nQ: next\nA: answer\n"
    result = parse_qa(raw)
    assert result.pairs[0].completion == "first line\n\nsecond paragraph"
    assert len(result.pairs) == 2


def test_repeated_lines_cleaning_keeps_qa_markers():
    raw = "Q: one?\nA: one\n\nQ: two?\nA: two\n\nQ: three?\nA: three\n"
    result = clean_text(raw, CleanOptions(remove_repeated_lines=True))
    assert len(parse_qa(result.text).pairs) == 3


def test_reflow_qa_pair_keeps_a_line():
    pair = QAPair("Q: A long\nquestion\nA:", "An answer\nsplit over lines.")
    out = reflow_qa_pair(pair)
    assert out.prompt == "Q: A long question\nA:"
    assert out.completion == "An answer split over lines."
