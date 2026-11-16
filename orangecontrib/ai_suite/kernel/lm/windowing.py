"""Window construction for next-token training pairs, in the token domain.

The input is a BOS-marked token stream - what ``Tokenizer.encode`` /
``Tokenizer.encode_corpus`` produce - so this module deals in integers only:
no text, no tokenizer, no vocabulary.

The stream is self-describing. A BOS before every sentence means sentence
structure travels inside the data, so windows cannot cross a sentence boundary
even by accident, and nothing here needs a parallel list of sentence lengths.

For a sentence with ids s1..sm and context size k, the stream carries
``[BOS, s1..sm]``. The window for target s_i is the last k tokens before it,
left-padded with BOS: the first window is all-BOS, and only from window k+1
onwards is the context fully real tokens. BOS is never a target, which is what
lets generation treat a sampled BOS as "stop".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .tokenizer import BOS_ID


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


def make_windows(stream: Sequence[int], context_size: int) -> Windows:
    """Build all next-token windows of a BOS-marked token stream.

    The stream is split on ``BOS_ID``; each piece is one sentence and is
    windowed independently, with BOS as the left pad. Every window of every
    sentence is known in advance - ``len(stream)`` minus one per sentence - so
    the result is preallocated.

    A stream that does not start with BOS, or that has two BOS in a row, is
    rejected: it can only come from something that did not encode a document.
    """
    if int(context_size) < 1:
        raise ValueError("context_size must be >= 1")
    context_size = int(context_size)

    ids = np.asarray(stream, dtype=np.int32)
    starts = np.flatnonzero(ids == BOS_ID)
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError(
            "stream must start with <bos>: BOS marks every sentence start, "
            "so this only accepts a stream from Tokenizer.encode"
        )
    ends = np.append(starts[1:], len(ids))
    lengths = ends - starts - 1
    if (lengths < 1).any():
        raise ValueError("stream has an empty sentence (two <bos> in a row)")

    contexts = np.full((len(ids) - len(starts), context_size), BOS_ID, dtype=np.int32)
    targets = np.empty(len(contexts), dtype=np.int32)
    row = 0
    for start, end in zip(starts, ends, strict=True):
        sentence = ids[start:end]  # begins with BOS by construction
        for i in range(1, len(sentence)):
            lo = max(0, i - context_size)
            contexts[row, context_size - (i - lo) :] = sentence[lo:i]
            targets[row] = sentence[i]
            row += 1

    return Windows(
        contexts=contexts,
        targets=targets,
        context_size=context_size,
        sentence_lengths=tuple(int(v) for v in lengths),
    )
