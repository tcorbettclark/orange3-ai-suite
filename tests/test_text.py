"""Plain-text cleaning, splitting and Q/A parsing."""

import pytest

from edu_ai_suite.text import (
    CleanOptions,
    QAPair,
    RowGranularity,
    clean_text,
    parse_qa,
    reflow_qa_pair,
    split_sentences,
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


def test_row_granularities():
    text = "First sentence. Second sentence.\n\nA new paragraph."
    assert split_text(text, RowGranularity.WHOLE_FILE) == (text,)
    assert split_text(text, RowGranularity.PARAGRAPH) == (
        "First sentence. Second sentence.",
        "A new paragraph.",
    )
    assert split_text("one\ntwo\n", RowGranularity.PARAGRAPH) == ("one\ntwo",)
    assert split_text(text, RowGranularity.SENTENCE) == (
        "First sentence.",
        "Second sentence.",
        "A new paragraph.",
    )


def test_split_sentences():
    assert split_sentences("Hi! You are... ok? Yes.") == ["Hi!", "You are...", "ok?", "Yes."]
    assert split_sentences("AB.\nCD.") == ["AB.", "CD."]  # newline = hard break
    assert split_sentences("no terminator here") == ["no terminator here"]
    assert split_sentences("What?! No.") == ["What?!", "No."]  # "?!" stays together
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []
    assert split_sentences("Dr. Who") == ["Dr.", "Who"]  # accepted limitation
    # the terminator needs whitespace after it, so numbers and domains survive
    assert split_sentences("3.14 is pi.") == ["3.14 is pi."]
    assert split_sentences("see example.com") == ["see example.com"]


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


def test_newline_keeps_joined_documents_apart():
    # A corpus is assembled outside the kernel by joining documents: a newline is
    # a hard break, so no sentence merges across the seam. A space would merge.
    documents = ["Chapter one ends here", "and chapter two begins. It runs on."]
    joined = split_sentences("\n".join(documents))
    assert joined == [
        "Chapter one ends here",
        "and chapter two begins.",
        "It runs on.",
    ]
    assert split_sentences(" ".join(documents))[0] == (
        "Chapter one ends here and chapter two begins."
    )


@pytest.mark.parametrize("granularity", list(RowGranularity))
def test_every_declared_granularity_is_accepted(granularity):
    assert split_text("one.\n\ntwo.", granularity)


def test_granularity_values_are_the_menu_labels():
    assert [granularity.value for granularity in RowGranularity] == [
        "Whole file",
        "Paragraph",
        "Sentence",
    ]


def test_every_granularity_value_is_distinct():
    values = [granularity.value for granularity in RowGranularity]
    assert len(set(values)) == len(values)


def test_a_string_is_not_a_granularity():
    with pytest.raises(ValueError, match="expected a RowGranularity"):
        split_text("one", "chapters")  # type: ignore[arg-type]


def test_rows_do_not_change_the_sentences_the_model_sees():
    # The table shape is not the model's unit: joining the rows and splitting
    # sentences again must give the same sentences whatever the granularity.
    from edu_ai_suite.lm import tokenizer_from_text

    text = "The cat sat. A dog barked.\n\nNothing else happened that day."
    tokenizer = tokenizer_from_text(text, "word")
    streams = {
        granularity: tokenizer.encode("\n".join(split_text(text, granularity)))
        for granularity in RowGranularity
    }
    assert len(set(map(tuple, streams.values()))) == 1


# ---- page numbers: what counts as one, what must survive ------------------

#: Lines that are a page number and nothing else.
PAGE_NUMBER_LINES = [
    "12",
    "  7  ",
    "- 42 -",
    "\u2014 9",
    "page 13",
    "Page 13 of 200",
    "p. 8",
    "13 / 200",
    "iv",
    "IV",
    "xii",
    "XII",
    "MMXXIV",
    "mix",  # a valid numeral (1009), so this is the documented loss
]

#: Lines that must be kept. The whitespace and bare-dash cases are the
#: regression: every part of the numeral grammar can match nothing, so a
#: pattern without a non-empty assertion eats every paragraph break in a book.
NOT_PAGE_NUMBER_LINES = [
    "",
    "   ",
    "\t",
    "-",
    "\u2014",
    " - ",
    "page",
    "page of 200",
    "did",
    "mid",
    "mind",
    "civic",
    "im",
    "chapter iv",
    "the 3rd of May",
    "12 monkeys",
]


@pytest.mark.parametrize("line", PAGE_NUMBER_LINES)
def test_a_page_number_line_is_removed(line):
    result = clean_text(f"before\n{line}\nafter\n", CleanOptions(remove_page_numbers=True))
    assert result.text == "before\nafter"
    assert result.removed_lines == 1


@pytest.mark.parametrize("line", NOT_PAGE_NUMBER_LINES)
def test_a_non_page_number_line_survives(line):
    result = clean_text(f"before\n{line}\nafter\n", CleanOptions(remove_page_numbers=True))
    assert result.text == f"before\n{line}\nafter"
    assert result.removed_lines == 0


@pytest.mark.parametrize("line", ["", "   ", "\t", "-", "\u2014", " - "])
def test_the_page_number_pattern_never_matches_an_empty_numeral(line):
    # Pinned at the pattern, not only at the outcome: a blank line is the
    # paragraph break, and the rule that removes numbers must not touch it.
    from edu_ai_suite.text._clean import _PAGE_NUMBER

    assert _PAGE_NUMBER.match(line) is None


def test_removing_page_numbers_keeps_paragraph_breaks():
    text = "para A line 1\npara A line 2\n\n\npara B line 1\n\n12\n\npara C\n"
    result = clean_text(text, CleanOptions(remove_page_numbers=True))
    assert result.text == "para A line 1\npara A line 2\n\n\npara B line 1\n\n\npara C"
    assert result.removed_lines == 1  # the page number, and nothing else


def test_reflow_after_removing_page_numbers_keeps_each_paragraph():
    # The end-to-end consequence of the two tests above: reflow splits on blank
    # lines, so eating them merges the whole document into one paragraph.
    text = (
        "Line one of para A\nand its second line.\n\n12\n\n"
        "Line one of para B\nand its second line.\n"
    )
    result = clean_text(text, CleanOptions(remove_page_numbers=True, reflow=True))
    assert result.text.split("\n\n") == [
        "Line one of para A and its second line.",
        "Line one of para B and its second line.",
    ]


def test_page_number_rule_keeps_roman_spelled_words():
    # Best-effort on purpose: the roman branch uses valid-numeral grammar and is
    # case-insensitive, so "iv"/"XII"/"mix" (a valid numeral, 1009) are taken as
    # page numbers, while "did", "mid" and "civic" - not valid numerals - stay.
    result = clean_text(
        "mix\ndid\nmind\ncivic\nim\nPage 13\niv\nXII\n",
        CleanOptions(remove_page_numbers=True),
    )
    assert result.text == "did\nmind\ncivic\nim"
