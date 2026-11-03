"""Prepare the teaching corpus: Gibbon, Decline and Fall, Volume One.

The LM teaching corpus (spec/tiny-lm-suite-design.md §7): single-domain,
prose, instantly recognisable ("the Roman Empire book"), **531 KB** — the
spec's sweet spot for CPU-minute training.

Derived from the CCEL plain-text edition of Gibbon (public domain), via the
already-cleaned Volume-One prose (headers/page furniture/footnote *text*
removed upstream). The residual artefacts removed here, with the regexes
recorded for reproduction:

- footnote reference digits attached to sentence-final punctuation
  ("Crassus.1" -> "Crassus.") — the digit was an inline note pointer, not prose;
- standalone enumerative roman markers ("II. The spirit ..." -> "The spirit ..."),
  Gibbon's ordered-list section labels inside paragraphs;
- digits immediately attached to words ("number347" -> "number").

Usage (the derivation is one-command reproducible from the cleaned source
file; the committed demo/gibbon_corpus.txt is the artifact):

    uv run python demo/fetch_gibbon.py [--src <decline_vol1_clean.txt>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_SRC = BASE.parent.parent / "lionel" / "epq" / "limerick-app" / "decline_vol1_clean.txt"
TARGET = BASE / "gibbon_corpus.txt"

_FOOTNOTE_AFTER_PERIOD = re.compile(r"\.(\d+)")
_ROMAN_MARKERS = re.compile(r"\b(?:II|III|IV|VI{0,2}|V{1,2}I{0,2}|IX|XI{1,2})\. (?=[A-Z])")
_ATTACHED_DIGITS = re.compile(r"(?<=[a-z]{2})\d{1,4}\b(?=|[,.;:?!])")
_MULTI_BLANK = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = _FOOTNOTE_AFTER_PERIOD.sub(".", text)
    text = _ROMAN_MARKERS.sub("", text)
    text = _ATTACHED_DIGITS.sub("", text)
    # paragraphs separated by blank lines; single newlines inside a paragraph
    # are joined (the source is one-paragraph-per-line, so nothing to do,
    # but normalise stray double spaces)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip() + "\n"


def stats(text: str) -> str:
    from nn_kernel.lm import Tokenizer

    char_tok = Tokenizer.from_corpus([text], "char")
    word_tok = Tokenizer.from_corpus([text], "word")
    sentences = len(re.findall(r"[.!?…]", text))
    return (f"{len(text):,} chars | sentences ~{sentences:,} | "
            f"char vocab {char_tok.vocab_size} | word vocab {word_tok.vocab_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    args = parser.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(
            f"source not found: {src}\n"
            "(derive decline_vol1_clean.txt first, or point --src elsewhere)"
        )
    raw = src.read_text(encoding="utf-8")
    text = clean(raw)
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    print("before:", len(raw), "chars; after:", len(text), "chars")
    try:
        print("corpus stats:", stats(text))
    except Exception as exc:  # kernel absent in bare envs — stats are optional
        print("(kernel stats skipped:", exc, ")")


if __name__ == "__main__":
    main()