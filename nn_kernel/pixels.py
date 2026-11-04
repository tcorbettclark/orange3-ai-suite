"""Raw pixel features: downsampled grayscale images as flat vectors.

This is the deliberately plain baseline: each image becomes a square grid of
luminance values in [0, 1], flattened row-major. It keeps no spatial structure
beyond the position of each value in the vector.
"""

from __future__ import annotations

import cv2
import numpy as np

from .features.preprocess import to_luminance


def _checked_size(size: int) -> int:
    size = int(size)
    if size < 1:
        raise ValueError("pixel grid size must be a positive integer")
    return size


def pixel_feature_names(size: int = 10) -> tuple[str, ...]:
    """Names of the flattened pixel features, in row-major order."""
    size = _checked_size(size)
    return tuple(f"pixel_r{r}_c{c}" for r in range(size) for c in range(size))


def pixel_features(image: np.ndarray, size: int = 10) -> np.ndarray:
    """Convert one image to a flat vector of ``size * size`` pixel values.

    The image is converted to luminance, scaled to [0, 1] when it is in 0-255
    range, resized to ``size x size`` and flattened row-major.
    """
    size = _checked_size(size)
    img = np.asarray(to_luminance(image), dtype=np.float32)
    if img.size == 0:
        raise ValueError("empty image")
    if float(img.max()) > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.shape != (size, size):
        shrinking = size < min(img.shape)
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=interpolation)
    return np.asarray(img, dtype=np.float32).reshape(-1)
