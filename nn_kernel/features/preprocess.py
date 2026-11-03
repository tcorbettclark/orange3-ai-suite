"""Preprocessing: luminance, centre crop, resize, denoise, contrast normalise.

All features are computed on the preprocessed float32 luminance image in
[0, 1]. Colour is deliberately dropped here (spec §2: "compute all features on
luminance only").

Pipeline order: luminance -> centre crop -> resize -> denoise ->
contrast-normalise. Denoising acts on raw scale (before normalisation);
normalisation is last so every feature sees the same [0, 1] range.
"""

from __future__ import annotations

import cv2
import numpy as np

from .options import FeatureOptions

_EPS = 1e-12

# ITU-R BT.601 luma weights, RGB channel order convention (the widget adapter
# is responsible for converting cv2's BGR before calling the kernel). Applied
# elementwise in to_luminance (kept for documentation/tests).
_LUMA = (0.299, 0.587, 0.114)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grayscale array to a float32 grayscale image.

    Accepts 2-D input as-is (already grayscale) and 3-D input with 3 (RGB,
    per this module's convention) or 4 (RGBA, alpha dropped) channels.
    Elementwise sums, deliberately NOT a matrix-vector matmul: some numpy
    builds (e.g. 2.1.3/macOS arm) emit spurious BLAS warnings on the `@`
    dispatch for mathematically identical operations.
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
            f"expected 1, 3 or 4 channels, got {img.shape[-1]} — pass RGB "
            "(convert cv2 BGR to RGB in the widget adapter before calling)"
        )
    r = img[..., 0].astype(np.float64)
    g = img[..., 1].astype(np.float64)
    b = img[..., 2].astype(np.float64)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _centre_crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _contrast_normalise(img: np.ndarray) -> np.ndarray:
    """Clip to the image's own [p1, p99] and scale to [0, 1].

    Adaptive by construction (spec §2: absolute thresholds silently encode
    dataset lighting). A flat image degenerates; we return all-zeros so that
    downstream features behave consistently for flat input.
    """
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < _EPS:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess_colour(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """Geometric pipeline only (centre-crop + resize), NO intensity ops:
    colour statistics need absolute channels, which contrast normalisation
    or median filtering would distort. RGB in [0, 1] or grayscale passthrough.
    (Used by the optional colour trap group only.)
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