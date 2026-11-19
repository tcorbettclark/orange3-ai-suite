"""The training pairs of a corpus: one context and one target each.

A pair is one prediction: the ``context_width`` ids the model reads, BOS-padded at
the start of a unit, and the id it must produce, which is never BOS. The input is
the EOS-terminated stream ``Tokenizer.encode`` produces, so the module deals in
integers only, and no pair straddles a unit boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .._holdout import held_out_positions
from .tokenizer import BOS_ID, EOS_ID, stream_units


@dataclass
class Pairs:
    """All (context, target) pairs of a corpus, one per prediction, unit-disjoint.

    ``contexts`` is shaped ``(len(targets), context_width)``, and every unit's last
    target is ``<eos>``, so unit boundaries are recoverable from ``targets``.
    """

    contexts: np.ndarray  # (N, w) int32
    targets: np.ndarray  # (N,) int32, one EOS per unit
    context_width: int

    def __post_init__(self) -> None:
        self.contexts = np.asarray(self.contexts, dtype=np.int32)
        self.targets = np.asarray(self.targets, dtype=np.int32)
        if self.contexts.shape != (len(self.targets), self.context_width):
            raise ValueError("contexts must be shaped (len(targets), context_width)")
        if len(self.targets) and int(self.targets[-1]) != EOS_ID:
            raise ValueError("targets must end with <eos>: every unit is terminated")

    def __len__(self) -> int:
        return len(self.targets)

    # ---- unit structure, derived -------------------------------------------

    def _offsets(self) -> np.ndarray:
        """Pair index where each unit starts, plus a trailing sentinel."""
        return np.concatenate(([0], np.flatnonzero(self.targets == EOS_ID) + 1))

    @property
    def pairs_per_unit(self) -> tuple[int, ...]:
        """Pairs contributed by each unit, in order: its tokens plus one EOS."""
        return tuple(int(b - a) for a, b in pairwise(self._offsets()))

    # ---- splitting ---------------------------------------------------------

    def split(self, validation_fraction: float = 0.1, seed: int = 0) -> tuple[Pairs, Pairs]:
        """Unit-level train/validation split: a stride over the units, from the seed.

        Held-out units are chosen by position, so the split is a function of ``seed`` and
        of corpus order alone: reordering or appending units can move a unit across, and
        units with identical ids may land on different sides.
        """
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        n = len(self._offsets()) - 1
        if n < 2:
            raise ValueError("need at least 2 units to split")

        is_val = np.zeros(n, dtype=bool)
        is_val[held_out_positions(n, validation_fraction, seed)] = True
        return self._gather(np.flatnonzero(~is_val)), self._gather(np.flatnonzero(is_val))

    def _gather(self, units: np.ndarray) -> Pairs:
        """A new Pairs holding just the given units, in the order given."""
        offsets = self._offsets()
        if len(units) == 0:
            return Pairs(
                contexts=np.empty((0, self.context_width), dtype=np.int32),
                targets=np.empty((0,), dtype=np.int32),
                context_width=self.context_width,
            )
        slices = [slice(offsets[u], offsets[u + 1]) for u in units]
        return Pairs(
            contexts=np.concatenate([self.contexts[s] for s in slices]),
            targets=np.concatenate([self.targets[s] for s in slices]),
            context_width=self.context_width,
        )


def make_pairs(stream: Sequence[int], context_width: int) -> Pairs:
    """Every pair of an EOS-terminated token stream, one per id in it.

    The stream must come from ``Tokenizer.encode``: one that does not end with
    ``<eos>``, that carries two ``<eos>`` in a row, or that carries ``<bos>`` is
    rejected. The result holds exactly ``len(stream)`` pairs, each id predicted once.
    """
    if int(context_width) != context_width or context_width < 1:
        raise ValueError("context_width must be a whole number >= 1")

    ids = np.asarray(stream, dtype=np.int32)
    if len(ids) == 0:
        raise ValueError("stream is empty: empty or whitespace-only text has no units")
    if BOS_ID in ids:
        raise ValueError(
            "stream carries <bos>: it opens a context and is never data, so this "
            "only accepts a stream from Tokenizer.encode"
        )
    # Unit structure is stream_units' to derive, and the shape of what it
    # returns is the whole validation: a non-empty last entry means the stream
    # did not end with EOS; an empty entry anywhere else is a doubled EOS.
    units = stream_units(ids)
    if units[-1]:
        raise ValueError(
            "stream must end with <eos>: every unit is terminated, so this only "
            "accepts a stream from Tokenizer.encode"
        )
    units.pop()  # the empty entry after the final EOS: the last pair predicts it
    if not all(units):
        raise ValueError("stream has an empty unit (two <eos> in a row)")

    contexts = np.full((len(ids), context_width), BOS_ID, dtype=np.int32)
    targets = np.empty(len(ids), dtype=np.int32)
    row = 0
    for unit in units:
        seq = np.concatenate(([BOS_ID], np.asarray(unit, dtype=np.int32), [EOS_ID]))
        for i in range(1, len(seq)):
            lo = max(0, i - context_width)
            contexts[row, context_width - (i - lo) :] = seq[lo:i]
            targets[row] = seq[i]
            row += 1

    return Pairs(contexts=contexts, targets=targets, context_width=context_width)
