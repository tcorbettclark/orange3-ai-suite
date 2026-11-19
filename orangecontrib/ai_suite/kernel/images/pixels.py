"""An image as a row of luminance values.

Decode to brightness, scale to [0, 1], resize to a square grid, flatten that grid
into a row.
"""

from __future__ import annotations

import numpy as np

from .decode import to_luminance, to_unit_scale
from .flatten import flatten
from .resize import resize_square


def pixel_row(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """One image as a flat row of ``size * size`` values in [0, 1], with names."""
    return flatten(resize_square(to_unit_scale(to_luminance(image)), size))
