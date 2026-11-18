"""Row granularity: how much text one row of the table holds.

The units the model trains on are always sentences - :func:`split_sentences`, and
``Tokenizer.encode`` closes each with EOS. This module only decides where table
rows end, which is a storage and selection choice, not a modelling one.
"""

from __future__ import annotations

import re
from enum import StrEnum


class RowGranularity(StrEnum):
    """How much text one table row holds.

    The member name is for code, the value is the label a reader sees in the
    menu, so there is one name to change rather than a member, a token and a
    caption kept in step. Sentences are the model's units whatever this says.
    """

    WHOLE_FILE = "Whole file"
    PARAGRAPH = "Paragraph"
    SENTENCE = "Sentence"


# Split after terminal punctuation followed by whitespace ("What?! No." keeps
# "?!" together). Newlines are hard breaks; lines without terminal punctuation
# are sentences of their own. Known limitation for teaching corpora:
# abbreviations such as "Mr." split. The whitespace requirement is why this has
# to run on raw text: word-level tokenization drops it, so "3.14" would
# otherwise be cut in half.
_TERMINATOR = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    """Split a document into sentences; a newline is a hard break.

    Text-level on purpose: the two rules it encodes - a newline ends a
    sentence, a terminator ends one only when whitespace follows - both depend
    on characters that tokenization may discard. Callers that work in the
    token domain get sentence structure from the tokenizer, which closes every
    sentence with EOS (see ``Tokenizer.encode``).

    Because a newline always breaks, documents joined by newlines cannot merge
    the last sentence of one into the first of the next - which is why a corpus
    can be assembled outside the kernel, with no separator convention to
    remember beyond "newline between documents".
    """
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _TERMINATOR.split(line):
            part = part.strip()
            if part:
                out.append(part)
    return out


def split_text(text: str, granularity: RowGranularity) -> tuple[str, ...]:
    """Cut one cleaned document into table rows of the selected granularity."""
    if granularity is RowGranularity.WHOLE_FILE:
        parts = [text.strip()]
    elif granularity is RowGranularity.PARAGRAPH:
        parts = [part.strip() for part in re.split(r"\n\s*\n", text)]
    elif granularity is RowGranularity.SENTENCE:
        parts = split_sentences(text)
    else:  # a member added without a branch, or a string passed as if it were one
        raise ValueError(f"expected a RowGranularity, got {granularity!r}")
    return tuple(part for part in parts if part)
