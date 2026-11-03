"""Sentence-aware window construction for next-token training pairs.

Windows never cross sentence or document boundaries: the model must not learn
to continue across unrelated sentences. BOS doubles as the context pad, so
early tokens with short context are trainable.

For a sentence with ids s1..sm and context size k, the sequence is
``[BOS, s1..sm]``. The window for target s_i is the last k tokens before it,
left-padded with BOS: window 1 is all-BOS, and only from window k+1 onwards is
the context fully real tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .tokenizer import BOS_ID, Tokenizer

# Split after terminal punctuation followed by whitespace ("What?! No." keeps
# "?!" together). Newlines are hard breaks; lines without terminal punctuation
# are sentences of their own. Known limitation for teaching corpora:
# abbreviations such as "Mr." split.
_TERMINATOR = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    """Split a document into sentences; a newline is a hard break."""
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


@dataclass
class Windows:
    """All (context, target) training pairs of a corpus, sentence-disjoint.

    ``sentence_lengths`` records how many windows each sentence contributed,
    in corpus order, so the train/validation split can happen at sentence
    level: no sentence contributes windows to both sets.
    """

    contexts: np.ndarray  # (N, k) int32
    targets: np.ndarray  # (N,) int32
    context_size: int
    sentence_lengths: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.targets)

    def __post_init__(self) -> None:
        if len(self.sentence_lengths) and sum(self.sentence_lengths) != len(self):
            raise ValueError("sentence_lengths do not sum to the number of windows")

    def split(self, val_fraction: float = 0.1, seed: int = 0) -> tuple[Windows, Windows]:
        """Sentence-level train/validation split, deterministic under a seed."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        n = len(self.sentence_lengths)
        if n < 2:
            raise ValueError("need at least 2 sentences to split")
        order = np.random.default_rng(seed).permutation(n)
        n_val = min(max(1, round(n * val_fraction)), n - 1)
        offsets = np.concatenate([[0], np.cumsum(self.sentence_lengths)])

        def gather(indices: np.ndarray) -> Windows:
            if len(indices) == 0:
                return Windows(
                    contexts=np.empty((0, self.context_size), dtype=np.int32),
                    targets=np.empty((0,), dtype=np.int32),
                    context_size=self.context_size,
                    sentence_lengths=(),
                )
            starts = offsets[indices]
            ends = offsets[indices + 1]
            lengths = (ends - starts).astype(int)
            return Windows(
                contexts=np.concatenate(
                    [self.contexts[s:e] for s, e in zip(starts, ends, strict=True)]
                ),
                targets=np.concatenate(
                    [self.targets[s:e] for s, e in zip(starts, ends, strict=True)]
                ),
                context_size=self.context_size,
                sentence_lengths=tuple(int(v) for v in lengths),
            )

        return gather(order[n_val:]), gather(order[:n_val])


def make_windows(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    context_size: int,
) -> Windows:
    """Tokenize documents and build all next-token windows.

    Sentences are split internally (:func:`split_sentences`); windows never
    span sentences. Empty sentences (no tokens after tokenization) are skipped
    and contribute no windows.
    """
    if int(context_size) < 1:
        raise ValueError("context_size must be >= 1")
    context_size = int(context_size)

    contexts: list[list[int]] = []
    targets: list[int] = []
    sentence_lengths: list[int] = []

    for text in texts:
        for sentence in split_sentences(text):
            ids = tokenizer.encode(sentence)
            if not ids:
                continue
            seq = [BOS_ID, *ids]
            for i in range(1, len(seq)):
                lo = max(0, i - context_size)
                window = [BOS_ID] * (context_size - (i - lo)) + seq[lo:i]
                contexts.append(window)
                targets.append(seq[i])
            sentence_lengths.append(len(ids))

    if not sentence_lengths:
        raise ValueError("no windows built: corpus has no non-empty sentences")

    return Windows(
        contexts=np.asarray(contexts, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32),
        context_size=context_size,
        sentence_lengths=tuple(sentence_lengths),
    )
