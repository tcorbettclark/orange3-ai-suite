"""Channel and value-range conventions shared by the image code.

Callers hand the kernel an array in the channel order and value range they have;
these helpers are the single place that says what that means. ``to_luminance``
converts RGB(A) or grayscale to luma, ``to_unit_scale`` brings pixel values into
[0, 1].

Both refuse rather than guess where guessing would silently change a feature: a
2-channel array is not RGB, and a 16-bit or HDR array is not a [0, 255] image.
"""

from __future__ import annotations

import numpy as np

# ITU-R BT.601 luma weights, RGB channel order. Callers must convert OpenCV's
# BGR first - the kernel never guesses channel order.
_LUMA = (0.299, 0.587, 0.114)
_MAX_8BIT = 255.0
_SCALE_TOLERANCE = 1e-6


def drop_alpha(image: np.ndarray) -> np.ndarray:
    """Drop a fourth (alpha) channel if there is one."""
    return image[..., :3] if image.ndim == 3 and image.shape[-1] == 4 else image


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grayscale array to a float32 grayscale image.

    Accepts 2-D input as-is (already grayscale) and 3-D input with 3 (RGB) or
    4 (RGBA, alpha dropped) channels. The weighted sum is elementwise rather
    than a matrix-vector product: some numpy builds emit spurious BLAS warnings
    for the equivalent matmul dispatch. Values keep the input's scale.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape}")
    image = drop_alpha(image)
    if image.shape[-1] != 3:
        raise ValueError(
            f"expected a 2-D array or a 3-D array with 3 (RGB) or 4 (RGBA) "
            f"channels, got shape {image.shape} - pass RGB (convert OpenCV BGR "
            "to RGB before calling)"
        )
    red = image[..., 0].astype(np.float64)
    green = image[..., 1].astype(np.float64)
    blue = image[..., 2].astype(np.float64)
    wr, wg, wb = _LUMA
    return (wr * red + wg * green + wb * blue).astype(np.float32)


def to_unit_scale(image: np.ndarray) -> np.ndarray:
    """Scale pixel values into [0, 1], accepting the unit or the 8-bit range.

    A peak at or below 1.0 is taken as already unit-scaled; a peak up to 255 is
    divided by 255 and clipped. Anything brighter is rejected: dividing a
    16-bit or HDR image by 255 would keep it above 1 and quietly flip every
    threshold that assumes unit scale.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image
    peak = float(image.max())
    if peak <= 1.0:
        return image
    if peak > _MAX_8BIT * (1.0 + _SCALE_TOLERANCE):
        raise ValueError(
            f"image values must be in [0, 1] or [0, 255], got max {peak:.0f}; "
            "convert 16-bit or HDR data before calling"
        )
    return np.clip(image / _MAX_8BIT, 0.0, 1.0)
