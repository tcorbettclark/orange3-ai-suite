"""Text hygiene: the rules that turn a raw document into clean text."""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def reflow(text: str) -> str:
    """Join wrapped lines into paragraphs and repair end-of-line hyphens.

    Shared with :mod:`._qa`, which reflows the completion side of a pair.
    """
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
        text = reflow(text)
    return CleanResult(text.strip(), removed, gutenberg_stripped)
