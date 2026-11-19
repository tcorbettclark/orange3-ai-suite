"""Which positions a validation split holds out: a stride over the corpus.

:meth:`lm.pairs.Pairs.split` and :meth:`lora.LoRAData.split` take this arithmetic
from here: about a fraction of the items, leaving at least one on each side whenever
there is anything to split.

The stride depends on corpus order: reordering a corpus can move an item from one
side of the split to the other, and a reproducible run re-reads the same corpus in
the same order.
"""

from __future__ import annotations


def held_out_positions(n: int, fraction: float, seed: int = 0) -> list[int]:
    """Positions of ``range(n)`` held out for validation, ``fraction`` of them.

    Every ``step``-th position from the offset ``seed % step``, where
    ``step = max(1, round(1 / fraction))``. The count is clamped to ``[1, n - 1]``, so
    both sides are non-empty whenever there is anything to split.

    ``fraction`` must be in ``(0, 1)``. ``n < 2`` has nothing to split and returns an
    empty list, the caller's cue to refuse or to skip validation. The seed only picks
    the stride's offset, so a corpus with fewer items than ``step`` has one possible
    split whatever the seed.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    if n < 2:
        return []
    step = max(1, round(1.0 / fraction))
    offset = seed % step
    positions = list(range(offset, n, step))
    if len(positions) > n - 1:
        positions = positions[: n - 1]
    return positions or [min(offset, n - 1)]
