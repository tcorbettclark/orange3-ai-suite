"""The training child's wire format: one typed event per marked stdout line.

:mod:`mlx_child` writes a line with :func:`encode` and :mod:`mlx_process` reads it
with :func:`decode`; a line the child does not mark is mlx-lm's own human log, which
the run keeps for the tail a failure message quotes. The reader and the writer import
this module and never each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Params:
    """How much of the model this run trains: a percentage and its raw counts."""

    fraction: float  # percent of parameters trained, as mlx-lm reports it
    detail: str  # e.g. "6.210M/2667.000M"


@dataclass(frozen=True)
class TrainReport:
    """One training report: the loss over the iterations since the last one."""

    iteration: int
    loss: float
    tokens_per_second: int


@dataclass(frozen=True)
class ValReport:
    """One validation report, on the same iteration axis as the training reports."""

    iteration: int
    loss: float


Event = Params | TrainReport | ValReport

#: Prefix of a structured event line. The one spelling, shared by both ends.
MARKER = "##mlx-metric "

#: The wire name of each event, and back again. One table, so a new event is a
#: row here and a class above - never a `kind` string appearing in two places.
_KINDS: dict[type, str] = {Params: "params", TrainReport: "train", ValReport: "val"}
_BY_KIND: dict[str, type] = {kind: cls for cls, kind in _KINDS.items()}


def encode(event: Event) -> str:
    """The line for one event, without its newline."""
    payload: dict[str, Any] = {"kind": _KINDS[type(event)], **asdict(event)}
    return MARKER + json.dumps(payload)


def decode(line: str) -> Event | None:
    """The event on this line, or None when the line is mlx-lm's human log.

    A malformed line, or a kind this version does not know, raises: both are bugs in our
    own child rather than noise to skip.
    """
    if not line.startswith(MARKER):
        return None
    try:
        payload = json.loads(line[len(MARKER) :])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the training child wrote a malformed event: {line!r}") from exc
    kind = payload.pop("kind", None)
    event = _BY_KIND.get(kind)
    if event is None:
        raise ValueError(f"the training child wrote an unknown event kind: {kind!r}")
    return event(**payload)
