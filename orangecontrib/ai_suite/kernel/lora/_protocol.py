"""The training child's wire format: one JSON metric per stdout line.

Both ends of this protocol are ours - :mod:`._child` writes with :func:`encode`,
:mod:`._protocol`'s reader decodes with :func:`decode` - so the format has one
author and no parser to reverse-engineer. It lives in its own module because the
reader (``training``) and the writer (``_child``, run with ``-m``) must be able to
import it without importing each other.

Every line the child does not mark is mlx-lm's own human log and is forwarded
untouched, so the tail a failure message quotes still says what happened.
"""

from __future__ import annotations

import json
from typing import Any

#: Prefix of a structured metric line. The one spelling, shared by both ends.
MARKER = "##mlx-metric "


def encode(metric: dict[str, Any]) -> str:
    """The line for one metric, without its newline."""
    return MARKER + json.dumps(metric)


def decode(line: str) -> dict[str, Any] | None:
    """The metric on this line, or None when the line is mlx-lm's human log.

    A malformed line is a bug in our own child, so it raises rather than being
    skipped as noise.
    """
    if not line.startswith(MARKER):
        return None
    try:
        return json.loads(line[len(MARKER) :])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the training child wrote a malformed metric: {line!r}") from exc
