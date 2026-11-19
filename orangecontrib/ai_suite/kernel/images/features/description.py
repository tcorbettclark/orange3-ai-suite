"""A feature described in words: title, summary, intuition, technical and references.

``title`` names the table column, ``summary`` is the one-sentence tooltip,
``intuition`` gives the plain-words reading of which pictures move the number,
``technical`` gives how this implementation computes it, and ``references`` holds
Markdown links to further reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LINK = re.compile(r"^\[[^\]]+\]\(https?://[^)\s]+\)$")
_SENTENCE_END = "."

#: Words that belong in ``technical``, not in ``intuition``.
JARGON = frozenset(
    {
        "covariance",
        "derivative",
        "distribution",
        "eigenvalue",
        "eigenvector",
        "gaussian",
        "histogram",
        "kernel",
        "matrix",
        "monotonic",
        "percentile",
        "standard deviation",
        "variance",
        "vector",
    }
)


@dataclass(frozen=True)
class Description:
    """One feature's four descriptions, checked on construction."""

    title: str
    summary: str
    intuition: str
    technical: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 3 <= len(self.title) <= 40:
            raise ValueError(
                f"title must be 3-40 characters, got {len(self.title)}: {self.title!r}"
            )
        if self.title.endswith(_SENTENCE_END):
            raise ValueError(f"title is a name, not a sentence: {self.title!r}")
        if not self.title[0].isupper():
            raise ValueError(f"title must start with a capital: {self.title!r}")
        if not 1 <= len(self.summary) <= 140:
            raise ValueError(f"summary must be 1-140 characters, got {len(self.summary)}")
        if not self.summary[0].isupper() or not self.summary.endswith(_SENTENCE_END):
            raise ValueError(f"summary must be one capitalised sentence: {self.summary!r}")
        if "\n" in self.summary:
            raise ValueError("summary must be a single line")
        if not 60 <= len(self.intuition) <= 600:
            raise ValueError(f"intuition must be 60-600 characters, got {len(self.intuition)}")
        if not 40 <= len(self.technical) <= 1400:
            raise ValueError(f"technical must be 40-1400 characters, got {len(self.technical)}")
        for link in self.references:
            if not _LINK.match(link):
                raise ValueError(f"reference must be a Markdown link to a URL: {link!r}")
        jargon = sorted(word for word in JARGON if word in self.intuition.lower())
        if jargon:
            raise ValueError(f"intuition uses technical vocabulary {jargon}; say it in plain words")
