"""Window construction for next-token training pairs, in the token domain.

The input is a BOS-marked token stream - what ``Tokenizer.encode`` /
``Tokenizer.encode_corpus`` produce - so this module deals in integers only:
no text, no tokenizer, no vocabulary.

The stream is self-describing: EOS closes every unit, so unit structure travels
inside the data and windows cannot cross a boundary even by accident. Nothing
here needs a parallel list of unit lengths.

For a unit with ids s1..sm and context size k, the windower builds
``[BOS, s1..sm, EOS]``. The window for each target is the last k tokens before
it, left-padded with BOS: the first window is all-BOS, and only from window k+1
onwards is the context fully real tokens. Targets are s1..sm followed by EOS,
so a unit contributes m+1 windows and the stream contributes exactly
``len(stream)`` of them. BOS is never a target - it is context construction,
not data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .tokenizer import BOS_ID, EOS_ID


@dataclass
class Windows:
    """All (context, target) training pairs of a corpus, unit-disjoint.

    ``sentence_lengths`` records how many windows each unit contributed, in
    corpus order, so the train/validation split can happen at unit level: no
    unit contributes windows to both sets. (The field name is historical - the
    unit is a sentence today; see the module docstring.)
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


def make_windows(stream: Sequence[int], context_size: int) -> Windows:
    """Build all next-token windows of an EOS-terminated token stream.

    The stream is split on ``EOS_ID``; each piece is one unit and is windowed
    independently, with BOS as the start-of-context marker and the left pad, and
    with EOS as its final target. Every token in the stream is predicted exactly
    once, so the result is preallocated to ``len(stream)`` rows.

    A stream that does not end with EOS, or that contains two EOS in a row, is
    rejected: it can only come from something that did not encode a document.
    """
    if int(context_size) < 1:
        raise ValueError("context_size must be >= 1")
    context_size = int(context_size)

    ids = np.asarray(stream, dtype=np.int32)
    ends = np.flatnonzero(ids == EOS_ID)
    if len(ends) == 0 or ends[-1] != len(ids) - 1:
        raise ValueError(
            "stream must end with <eos>: every unit is terminated, so this only "
            "accepts a stream from Tokenizer.encode"
        )
    starts = np.concatenate(([0], ends[:-1] + 1))
    if (ends - starts < 1).any():
        raise ValueError("stream has an empty unit (two <eos> in a row)")

    contexts = np.full((len(ids), context_size), BOS_ID, dtype=np.int32)
    targets = np.empty(len(ids), dtype=np.int32)
    row = 0
    for start, end in zip(starts, ends, strict=True):
        # BOS opens the unit's context; EOS is its last target.
        seq = np.concatenate(([BOS_ID], ids[start:end], [EOS_ID]))
        for i in range(1, len(seq)):
            lo = max(0, i - context_size)
            contexts[row, context_size - (i - lo) :] = seq[lo:i]
            targets[row] = seq[i]
            row += 1

    return Windows(
        contexts=contexts,
        targets=targets,
        context_size=context_size,
        sentence_lengths=tuple(int(v) + 1 for v in ends - starts),
    )
