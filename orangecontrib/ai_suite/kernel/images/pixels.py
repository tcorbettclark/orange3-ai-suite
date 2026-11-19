"""The plain baseline: one image becomes a small grid of luminance values.

Decode, resize, flatten - in that order and nothing else: no features, no table.
This is the one spelling of the plain pixel pipeline, so the widget that shows it
and the example that prints it cannot drift apart, and each step keeps its own
module and its own tests.
"""

from __future__ import annotations

import numpy as np

from .decode import to_luminance, to_unit_scale
from .flatten import flatten
from .resize import resize_square


def pixel_row(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """One image as a flat row of ``size * size`` values in [0, 1], with names.

    The order is the content of this function, and each step is there for a
    reason: brightness before resampling, so the resampler averages luminance
    rather than channel values that only later become luminance; unit scale
    before resampling, so the grid means the same thing for a dark and a bright
    photograph; resize before flatten, so the row is square whatever the source's
    aspect. The names come from the flattened grid's own shape
    (:func:`~.flatten.flatten`), so they cannot disagree with the values.
    """
    return flatten(resize_square(to_unit_scale(to_luminance(image)), size))
