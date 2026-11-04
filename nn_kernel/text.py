"""Plain-text helpers: cleaning, row splitting and Q/A parsing.

This module is deliberately independent of Orange and Qt. It works on strings
and returns immutable results so the widget layer only has to handle files,
previews and Tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lm.windowing import split_sentences

_PG_START = re.compile(
    r"^\s*\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE
)
_PG_END = re.compile(
    r"^\s*\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*$", re.IGNORECASE | re.MULTILINE
)
_PAGE_NUMBER = re.compile(
    r"^\s*[-–—]?\s*(?:(?:page|p\.)\s*)?"
    r"(?:\d{1,4}|[ivxlcdm]{1,7})"
    r"(?:\s*(?:of|/)\s*\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
_REPEAT_LIMIT = 80


@dataclass(frozen=True)
class CleanOptions:
    """Opt-in cleaning rules for plain text."""

    strip_gutenberg: bool = False
    reflow: bool = False
    remove_repeated_lines: bool = False
    remove_page_numbers: bool = False


@dataclass(frozen=True)
class CleanResult:
    text: str
    removed_lines: int = 0
    gutenberg_stripped: bool = False


@dataclass(frozen=True)
class QAPair:
    prompt: str
    completion: str


@dataclass(frozen=True)
class QAParseResult:
    pairs: tuple[QAPair, ...]
    skipped: int
    preamble_lines: int


def _normalise(text: str) -> str:
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _strip_gutenberg(text: str) -> tuple[str, bool]:
    """Keep the body between the standard Gutenberg marker lines."""
    stripped = False
    start = _PG_START.search(text)
    if start is not None:
        text = text[start.end() :]
        stripped = True
    end = _PG_END.search(text)
    if end is not None:
        text = text[: end.start()]
        stripped = True
    return text, stripped


def _reflow(text: str) -> str:
    """Join wrapped lines into paragraphs and repair end-of-line hyphens."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        paragraph = " ".join(lines)
        paragraph = re.sub(r"(\w)-\s+(\w)", r"\1\2", paragraph)
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _remove_page_numbers(text: str) -> tuple[str, int]:
    kept = [line for line in text.splitlines() if not _PAGE_NUMBER.match(line)]
    return "\n".join(kept), len(text.splitlines()) - len(kept)


def _remove_repeated_lines(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    counts: dict[str, int] = {}
    for line in lines:
        key = line.strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, count in counts.items() if count >= 3 and len(key) <= _REPEAT_LIMIT}
    kept = [line for line in lines if line.strip() not in repeated]
    return "\n".join(kept), len(lines) - len(kept)


def clean_text(text: str, options: CleanOptions | None = None) -> CleanResult:
    """Apply the selected cleaning rules in a stable order."""
    opts = options or CleanOptions()
    text = _normalise(text)
    removed = 0
    gutenberg_stripped = False
    if opts.strip_gutenberg:
        text, gutenberg_stripped = _strip_gutenberg(text)
    if opts.remove_page_numbers:
        text, n = _remove_page_numbers(text)
        removed += n
    if opts.remove_repeated_lines:
        text, n = _remove_repeated_lines(text)
        removed += n
    if opts.reflow:
        text = _reflow(text)
    return CleanResult(text.strip(), removed, gutenberg_stripped)


def split_text(text: str, mode: str) -> tuple[str, ...]:
    """Split one cleaned document into the selected row unit."""
    if mode == "whole":
        parts = [text.strip()]
    elif mode == "paragraphs":
        parts = [part.strip() for part in re.split(r"\n\s*\n", text)]
    elif mode == "lines":
        parts = [line.strip() for line in text.splitlines()]
    elif mode == "sentences":
        parts = split_sentences(text)
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    return tuple(part for part in parts if part)


def _is_q(line: str) -> bool:
    return line.lstrip().startswith("Q:")


def _is_a(line: str) -> bool:
    return line.lstrip().startswith("A:")


def parse_qa(text: str) -> QAParseResult:
    """Parse fixed Q:/A: blocks into prompt/completion pairs.

    A new pair starts at any line beginning with ``Q:``. The prompt includes
    the ``A:`` marker line; the completion runs until the next ``Q:`` line.
    """
    lines = text.splitlines()
    pairs: list[QAPair] = []
    skipped = 0
    preamble = 0
    i = 0
    seen_q = False
    while i < len(lines):
        if not _is_q(lines[i]):
            if not seen_q and lines[i].strip():
                preamble += 1
            i += 1
            continue
        seen_q = True
        q_start = i
        j = i + 1
        while j < len(lines) and not _is_a(lines[j]) and not _is_q(lines[j]):
            j += 1
        if j >= len(lines) or not _is_a(lines[j]):
            skipped += 1
            i = j
            continue
        a_line = j
        a_text = lines[a_line]
        a_col = a_text.find("A:")
        k = j + 1
        while k < len(lines) and not _is_q(lines[k]):
            k += 1
        prompt = "\n".join([*lines[q_start:a_line], a_text[: a_col + 2].strip()]).strip()
        completion = "\n".join([a_text[a_col + 2 :].strip(), *lines[a_line + 1 : k]]).strip()
        if prompt and completion:
            pairs.append(QAPair(prompt, completion))
        else:
            skipped += 1
        i = k
    return QAParseResult(tuple(pairs), skipped, preamble)


def reflow_qa_pair(pair: QAPair) -> QAPair:
    """Reflow a Q/A pair without merging the prompt and its ``A:`` marker."""
    prompt_lines = pair.prompt.splitlines()
    a_index = next(
        (i for i, line in enumerate(prompt_lines) if _is_a(line)),
        len(prompt_lines) - 1,
    )
    q_text = " ".join(line.strip() for line in prompt_lines[:a_index] if line.strip())
    a_text = prompt_lines[a_index].strip() if a_index < len(prompt_lines) else "A:"
    prompt = f"{q_text}\n{a_text}".strip()
    completion = _reflow(pair.completion)
    return QAPair(prompt, completion)
