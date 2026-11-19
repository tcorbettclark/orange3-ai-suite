"""The record one training epoch produces: its fields, and their order.

It pulls in nothing heavier than ``typing``, so a widget can read the key order without
the ML stack. ``EpochStats`` is a plain dict's schema: an ``on_epoch`` hook may add keys
of its own, and reordering the annotations reorders the table's columns.
"""

from __future__ import annotations

from typing import TypedDict


class EpochStats(TypedDict, total=False):
    """One epoch's numbers, plus whatever an ``on_epoch`` hook adds."""

    epoch: int
    loss: float
    val_loss: float
    perplexity: float
    val_perplexity: float


#: The keys :func:`~.training.epoch_callback` always fills, in the order it fills
#: them; the widget-side checkpoint table reads them as its columns, in that
#: order. Derived from :class:`EpochStats`, never re-listed.
EPOCH_STATS_KEYS: tuple[str, ...] = tuple(EpochStats.__annotations__)
