"""Which positions a validation split holds out: a stride over the corpus.

:meth:`lm.pairs.Pairs.split` and :meth:`lora.LoRAData.split` make the same
mechanical decision - take about a fraction of the items, leaving at least one
on each side - so the arithmetic is spelled once here. What stays with the
callers is their own policy, which genuinely differs: units versus rows, and a
fraction that must exclude 0 (a corpus with no training units is not a run)
versus one that allows it (no validation set at all is a mode).

The rule is a stride, so it depends on corpus order - deliberately. Nothing
here hashes content: reordering the corpus can move an item from one side of the
split to the other, so a reproducible run re-reads the same corpus in the same
order. That is a weaker promise than a content hash makes, and a much shorter
one to state.
"""

from __future__ import annotations


def held_out_positions(n: int, fraction: float, seed: int = 0) -> list[int]:
    """Positions of ``range(n)`` held out for validation, ``fraction`` of them.

    Every ``step``-th position starting at ``seed % step``, where
    ``step = max(1, round(1 / fraction))``. A stride spreads the held-out items
    across the corpus instead of taking one end of it, which for a book would
    make the validation set a different chapter. The count is clamped to
    ``[1, n - 1]``, so both sides are non-empty whenever there is anything to
    split at all.

    ``fraction`` must be in ``(0, 1)``; ``n < 2`` has nothing to split and
    returns an empty list, which is the caller's cue to refuse or to skip
    validation. The seed only picks the stride's offset, so a corpus with fewer
    items than ``step`` has exactly one possible split whatever the seed.
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
