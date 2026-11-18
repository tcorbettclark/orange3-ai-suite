"""Resampling an image to a fixed square resolution.

One policy for every consumer, so the two things that resample cannot drift apart:
area averaging when shrinking, bilinear when growing, decided per axis (a
non-square source can shrink on one axis while growing on the other). The frame
is stretched to a square, never cropped - cropping is a choice of whoever wants
it, and the feature pipeline makes it before calling here.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resample ``image`` to ``size x size``, keeping the whole frame.

    Accepts 2-D or 3-D input; a third dimension is treated as channels and
    resampled with the same policy. The output is float32 and contiguous.

    Area averaging when shrinking is the only correct choice for downsampling -
    it averages the pixels each cell covers instead of skipping them. Bilinear
    when growing avoids the blocky replication that area averaging produces when
    zooming in. Each axis gets its own decision, so a mixed source (smaller than
    the target on one axis, larger on the other) is downsampled first, then
    upsampled.
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
