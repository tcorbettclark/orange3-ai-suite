"""What one epoch of training reports: the record, and its key order.

A leaf module on purpose. The producer (:mod:`training`) and the consumer
(:mod:`orangecontrib.ai_suite.adapters`, which builds the checkpoint table) both
need the key order, but the producer imports Keras - so importing the keys must
not be a way to import the ML stack. The widget layer reads this module, not
:mod:`training`.

The record stays a plain dict rather than a frozen object, because a caller's
``on_epoch`` hook adds keys of its own (a sample of the epoch's generation, say)
and the table carries them alongside. The names are a :class:`TypedDict` so that
constructing a record is by name, never by position: reordering the annotations
reorders the columns, and cannot silently move a value to another key.
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
