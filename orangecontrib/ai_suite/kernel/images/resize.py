"""An image resampled to a fixed square, the whole frame stretched, never cropped.

Area averaging when an axis shrinks, bilinear when it grows, decided per axis.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resample ``image`` to ``size x size``, stretching the whole frame.

    Accepts 2-D or 3-D input, a third dimension treated as channels. The output
    is float32 and contiguous.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        raise ValueError("empty image")
    if image.shape[:2] == (size, size):
        return np.ascontiguousarray(image)
    # Each axis gets its own rule; cv2.resize's dsize is (width, height).
    shrinking = tuple(size < axis for axis in image.shape[:2])
    if shrinking[0] == shrinking[1]:
        interpolation = cv2.INTER_AREA if shrinking[0] else cv2.INTER_LINEAR
        return cv2.resize(np.ascontiguousarray(image), (size, size), interpolation=interpolation)
    # Mixed source: one axis shrinks, the other grows. Downsample first (area
    # averaging), then grow what is left (bilinear) - exactly one axis each way.
    image = np.ascontiguousarray(image)
    if shrinking[0]:
        image = cv2.resize(image, (image.shape[1], size), interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image, (size, image.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
