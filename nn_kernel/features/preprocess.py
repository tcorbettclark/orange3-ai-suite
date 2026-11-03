"""Preprocessing: luminance, centre crop, resize, denoise, contrast normalise.

Features are computed on a float32 luminance image in [0, 1]. Colour is dropped
deliberately: colour statistics belong to the opt-in shortcut group G, not to
the default feature set.

Pipeline order: luminance -> centre crop -> resize -> denoise ->
contrast-normalise. Denoising acts on the raw scale, before normalisation; the
normalisation is last so every feature sees the same [0, 1] range.
"""

from __future__ import annotations

import cv2
import numpy as np

from .options import FeatureOptions

_EPS = 1e-12

# ITU-R BT.601 luma weights, RGB channel order. The widget adapter converts
# OpenCV's BGR before calling the kernel.
_LUMA = (0.299, 0.587, 0.114)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grayscale array to a float32 grayscale image.

    Accepts 2-D input as-is (already grayscale) and 3-D input with 3 (RGB) or
    4 (RGBA, alpha dropped) channels. The weighted sum is elementwise rather
    than a matrix-vector product: some numpy builds emit spurious BLAS warnings
    for the equivalent matmul dispatch.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise ValueError(
            f"expected 1, 3 or 4 channels, got {img.shape[-1]} - pass RGB "
            "(convert OpenCV BGR to RGB in the widget adapter before calling)"
        )
    r = img[..., 0].astype(np.float64)
    g = img[..., 1].astype(np.float64)
    b = img[..., 2].astype(np.float64)
    wr, wg, wb = _LUMA
    return (wr * r + wg * g + wb * b).astype(np.float32)


def _centre_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _contrast_normalise(img: np.ndarray) -> np.ndarray:
    """Clip to the image's own [p1, p99] and scale to [0, 1].

    Adaptive by construction: absolute thresholds would silently encode the
    dataset's lighting. A flat image degenerates to all zeros so downstream
    features behave consistently.
    """
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < _EPS:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess_colour(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """Geometric pipeline only (centre-crop + resize), no intensity ops.

    Colour statistics need absolute channels, which contrast normalisation or
    median filtering would distort. Returns RGB in [0, 1] or grayscale
    passthrough; used by the opt-in colour group only.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
    if options.centre_crop:
        img = _centre_crop_to_square(img)
    if options.working_size is not None and img.shape != (options.working_size,) * 2:
        img = cv2.resize(
            img,
            (options.working_size, options.working_size),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(img, dtype=np.float32)


def preprocess(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """Run the full preprocessing pipeline; returns float32 in [0, 1]."""
    img = to_luminance(image)
    if options.centre_crop:
        img = _centre_crop_to_square(img)
    if options.working_size is not None and img.shape != (options.working_size,) * 2:
        img = cv2.resize(
            img,
            (options.working_size, options.working_size),
            interpolation=cv2.INTER_AREA,
        )
    if options.denoise:
        k = max(3, min(int(options.denoise_ksize), 5))
        if k % 2 == 0:
            k += 1
        img = cv2.medianBlur(np.ascontiguousarray(img, dtype=np.float32), k)
    if options.contrast_normalise:
        img = _contrast_normalise(img)
    return np.ascontiguousarray(img, dtype=np.float32)
