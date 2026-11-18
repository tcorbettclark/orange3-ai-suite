"""One image resampled to a fixed resolution: the raw pixel baseline.

``values.reshape(size, size)`` is the picture again at that resolution: one luma
value per cell, in [0, 1] (see :func:`decode.to_luminance`). The resolution is
fixed by the caller, never by the photograph, so every image yields the same
number of values.

The vector is flat, row-major, left to right then top to bottom: one image is one
row of a table and one grid cell is one column. Naming those columns is the
caller's job - they are coordinates, not concepts, and the kernel never uses them.

Mechanically this is resampling - area averaging when shrinking, bilinear when
growing. It commits to three choices that a plain ``cv2.resize`` leaves to
whoever calls it:

* square output - the frame is stretched, never cropped, so the aspect ratio is
  lost;
* a fixed grid, not a fixed scale - every image becomes the same number of
  values whatever its resolution;
* values in [0, 1] from the input's own scale - a dark photograph stays dark,
  unlike the catalogue pipeline, which contrast-normalises every image.

These are not catalogue features: the output is the picture itself, not
measurements of it, and nothing here is described in the feature guide. The only
thing shared with the features is input decoding, :mod:`decode`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .decode import to_luminance, to_unit_scale


def _checked_size(size: int) -> int:
    size = int(size)
    if size < 1:
        raise ValueError("pixel grid size must be a positive integer")
    return size


def pixel_values(image: np.ndarray, size: int) -> np.ndarray:
    """Convert one image to a flat vector of ``size * size`` pixel values.

    The resolution is required: a default would decide, invisibly, how much
    detail every caller keeps and how many columns a model is trained on.

    Luma is taken first, then values are brought to [0, 1] by
    :func:`decode.to_unit_scale`, which refuses ranges it cannot
    interpret (16-bit, HDR, negative) instead of saturating them to white or
    black. Resolution is left to this module: ``INTER_AREA`` when shrinking,
    ``INTER_LINEAR`` when growing.
    """
    size = _checked_size(size)
    img = np.asarray(to_luminance(image), dtype=np.float32)
    if img.size == 0:
        raise ValueError("empty image")
    img = to_unit_scale(img)
    if img.shape != (size, size):
        shrinking = size < min(img.shape)
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=interpolation)
    return np.asarray(img, dtype=np.float32).reshape(-1)
