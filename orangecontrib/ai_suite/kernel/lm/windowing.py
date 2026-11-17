"""Window construction for next-token training pairs, in the token domain.

The input is an EOS-terminated token stream - what ``Tokenizer.encode``
produces - so this module deals in integers only: no text, no tokenizer, no
vocabulary.

The stream is self-describing: EOS closes every unit, so unit structure travels
inside the data and windows cannot cross a boundary even by accident. Nothing
here needs a parallel list of unit lengths; they are derived from the EOS
targets in :attr:`Windows.unit_lengths`.

For a unit with ids s1..sm and context size k, the windower builds
``[BOS, s1..sm, EOS]``. The window for each target is the last k tokens before
it, left-padded with BOS: the first window is all-BOS, and only from window k+1
onwards is the context fully real tokens. Targets are s1..sm followed by EOS,
so a unit contributes m+1 windows and the stream contributes exactly
``len(stream)`` of them. BOS is never a target - it is context construction,
not data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .tokenizer import BOS_ID, EOS_ID


@dataclass
class Windows:
    """All (context, target) training pairs of a corpus, unit-disjoint.

    Unit structure is not stored: every unit's last window predicts EOS, so the
    boundaries are recoverable from ``targets`` (see :attr:`unit_lengths`).
    That keeps one source of truth, and makes an inconsistent object
    unrepresentable rather than merely validated.
    """

    contexts: np.ndarray  # (N, k) int32
    targets: np.ndarray  # (N,) int32, one EOS per unit
    context_size: int

    def __post_init__(self) -> None:
        self.contexts = np.asarray(self.contexts, dtype=np.int32)
        self.targets = np.asarray(self.targets, dtype=np.int32)
        if self.contexts.shape != (len(self.targets), self.context_size):
            raise ValueError("contexts must be shaped (n_windows, context_size)")
        if len(self.targets) and int(self.targets[-1]) != EOS_ID:
            raise ValueError("targets must end with <eos>: every unit is terminated")

    def __len__(self) -> int:
        return len(self.targets)

    # ---- unit structure, derived -------------------------------------------

    def _offsets(self) -> np.ndarray:
        """Window index where each unit starts, plus a trailing sentinel."""
        return np.concatenate(([0], np.flatnonzero(self.targets == EOS_ID) + 1))

    @property
    def unit_lengths(self) -> tuple[int, ...]:
        """Windows contributed by each unit, in order (tokens + its EOS)."""
        return tuple(int(b - a) for a, b in pairwise(self._offsets()))

    def _content_keys(self, seed: int) -> list[bytes]:
        """One salted hash per unit, computed from the unit's own ids."""
        salt = np.uint64(seed).tobytes()
        return [
            hashlib.blake2b(salt + self.targets[a:b].tobytes(), digest_size=8).digest()
            for a, b in pairwise(self._offsets())
        ]

    # ---- splitting ---------------------------------------------------------

    def split(self, val_fraction: float = 0.1, seed: int = 0) -> tuple[Windows, Windows]:
        """Unit-level train/validation split, deterministic under a seed.

        Units are selected by a salted hash of their own ids, in key order, so
        the split is a function of ``(seed, the set of units)`` - never of the
        order the corpus arrived in. Reordering or re-fetching documents cannot
        change it, and appending data does not move existing assignments.

        Selection is atomic per key: two units with identical ids hash alike and
        are therefore always assigned together, so a sentence can never appear
        in both train and validation. Atomicity can overshoot ``val_fraction``
        when a large group is taken, and if *every* unit is identical the ratio
        wins instead - duplicates then straddle the split, which is unavoidable
        when there is only one distinct unit in the corpus.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        n = len(self._offsets()) - 1
        if n < 2:
            raise ValueError("need at least 2 units to split")

        groups: dict[bytes, list[int]] = {}
        for index, key in enumerate(self._content_keys(seed)):
            groups.setdefault(key, []).append(index)

        target = min(max(1, round(n * val_fraction)), n - 1)
        val_units: list[int] = []
        for key in sorted(groups):
            group = groups[key]
            if len(val_units) >= target or len(val_units) + len(group) >= n:
                break  # enough held out, and training keeps at least one unit
            val_units.extend(group)
        if not val_units:
            # Every unit is identical: keeping duplicates apart would leave a
            # one-unit validation set, which is useless. Honour the ratio and
            # accept the overlap - it is unavoidable here.
            val_units = list(groups[sorted(groups)[0]][:target])

        is_val = np.zeros(n, dtype=bool)
        is_val[val_units] = True
        return self._gather(np.flatnonzero(~is_val)), self._gather(np.flatnonzero(is_val))

    def _gather(self, units: np.ndarray) -> Windows:
        """A new Windows holding just the given units, in the order given."""
        offsets = self._offsets()
        if len(units) == 0:
            return Windows(
                contexts=np.empty((0, self.context_size), dtype=np.int32),
                targets=np.empty((0,), dtype=np.int32),
                context_size=self.context_size,
            )
        slices = [slice(offsets[u], offsets[u + 1]) for u in units]
        return Windows(
            contexts=np.concatenate([self.contexts[s] for s in slices]),
            targets=np.concatenate([self.targets[s] for s in slices]),
            context_size=self.context_size,
        )


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

    return Windows(contexts=contexts, targets=targets, context_size=context_size)
