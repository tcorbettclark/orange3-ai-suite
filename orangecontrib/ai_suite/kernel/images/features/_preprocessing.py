"""The preprocessing pipeline: crop, resize, denoise, contrast-normalise.

Features are computed on a float32 luminance image in the order luminance ->
centre crop -> resize -> denoise -> contrast-normalise, with denoising on the raw
scale before the normalisation. Channel order and value scale are
:mod:`..decode`'s; :func:`extraction.extract_features` is the caller, picking the
pipeline each feature reads.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..decode import drop_alpha, to_luminance, to_unit_scale
from ..resize import resize_square
from ._conventions import EPS
from .options import FeatureOptions

__all__ = ["preprocess", "preprocess_colour"]


def _centre_crop_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def _geometry(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """The geometric half of both pipelines: centre-crop, then resize.

    The feature definitions assume the subject is central. Resampling is
    :mod:`..resize`.
    """
    if options.centre_crop:
        image = _centre_crop_to_square(image)
    if options.working_size is not None:
        image = resize_square(image, options.working_size)
    return image


def _contrast_normalise(image: np.ndarray) -> np.ndarray:
    """Clip to the image's own [p1, p99] and scale to [0, 1].

    A flat image degenerates to all zeros.
    """
    lo, hi = np.percentile(image, 1), np.percentile(image, 99)
    if hi - lo < EPS:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess_colour(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """Geometric pipeline only (centre-crop + resize), with values brought to [0, 1].

    Returns float32 RGB or grayscale, for the opt-in colour features only.
    """
    image = to_unit_scale(drop_alpha(np.asarray(image)))
    return np.ascontiguousarray(_geometry(image, options), dtype=np.float32)


def preprocess(image: np.ndarray, options: FeatureOptions) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Returns float32 in [0, 1] when ``contrast_normalise`` is on, and in the
    input's own intensity scale when it is off.
    """
    image = _geometry(to_luminance(image), options)
    if options.denoise:
        image = cv2.medianBlur(np.ascontiguousarray(image, dtype=np.float32), options.denoise_ksize)
    if options.contrast_normalise:
        image = _contrast_normalise(image)
    return np.ascontiguousarray(image, dtype=np.float32)
