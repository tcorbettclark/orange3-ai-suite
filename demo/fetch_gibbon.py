"""Prepare the language-model teaching corpus: Gibbon, Volume One.

Downloads the public-domain text of *The History of the Decline and Fall of the
Roman Empire* from Project Gutenberg and keeps the prose of Volume One. The
result is a single-domain corpus with an instantly recognisable style: ideal
for watching a small model move from gibberish to period pastiche.

The cleaner removes the Project Gutenberg header and footer, the table of
contents, footnotes and editor notes, and reflows the wrapped lines into
paragraphs. It is deterministic, so re-running reproduces the same file.

    uv run python demo/fetch_gibbon.py
    uv run python demo/fetch_gibbon.py --src ~/Downloads/gibbon.txt
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

SOURCE_URL = "https://www.gutenberg.org/ebooks/25717.txt.utf-8"
SOURCE_LABEL = "Project Gutenberg eBook #25717"

BASE = Path(__file__).resolve().parent
TARGET = BASE / "gibbon_corpus.txt"

_PG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_PG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"
# Volume One prose starts at the Introduction and stops where Volume Two
# begins. Both markers carry the book's six-space paragraph indent, which
# distinguishes them from the table-of-contents entries.
_VOLUME_ONE_START = "\n      Introduction.\n"
_VOLUME_TWO_START = "\n      VOLUME TWO\n"

# A footnote or editor note paragraph: "[ ... ]" or "12 (return) [ ... ]",
# possibly with the bracket on the following line.
_FOOTNOTE_START = re.compile(r"^\s*(?:\d+[a-z]?\s*\]?\s*)?\(\s*return\s*\)")
_NOTE_START = re.compile(r"^\s*\[\s")


def download(url: str = SOURCE_URL) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "orange3-nn-suite teaching-corpus fetcher"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def _strip_notes(text: str) -> str:
    """Drop footnote and editor-note paragraphs."""
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if not skipping and (_FOOTNOTE_START.match(line) or _NOTE_START.match(line)):
            skipping = True
            continue
        if skipping:
            if line.rstrip().endswith("]"):
                skipping = False
            continue
        kept.append(line)
    if skipping:
        raise ValueError("unterminated footnote or editor note in the source text")
    return "\n".join(kept)


def _paragraphs(text: str) -> str:
    """Reflow wrapped lines into paragraphs and repair end-of-line hyphens."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        paragraph = " ".join(lines)
        paragraph = re.sub(r"(\w)-\s+(\w)", r"\1\2", paragraph)
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs) + "\n"


def clean(raw: str) -> str:
    """Turn the Gutenberg text into the Volume One teaching corpus."""
    raw = raw.replace("\r\n", "\n")
    try:
        start = raw.index(_PG_START)
        end = raw.index(_PG_END)
    except ValueError as exc:
        raise ValueError("not a Project Gutenberg text: header/footer not found") from exc
    text = raw[start:end]
    try:
        first = text.index(_VOLUME_ONE_START)
        last = text.index(_VOLUME_TWO_START)
    except ValueError as exc:
        raise ValueError("Volume One markers not found in the Gutenberg text") from exc
    if last <= first:
        raise ValueError("Volume One markers are out of order")
    return _paragraphs(_strip_notes(text[first:last]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", help="use a local copy of the Gutenberg text")
    parser.add_argument("--out", default=str(TARGET), help="output corpus path")
    args = parser.parse_args()

    raw = Path(args.src).read_text(encoding="utf-8") if args.src else download()
    corpus = clean(raw)
    out = Path(args.out)
    out.write_text(corpus, encoding="utf-8")

    print(f"source: {SOURCE_LABEL}")
    print(
        f"wrote {out} ({len(corpus):,} chars, {corpus.count(chr(10) + chr(10)) + 1:,} paragraphs)"
    )


if __name__ == "__main__":
    main()
