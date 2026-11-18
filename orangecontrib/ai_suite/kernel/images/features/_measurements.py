"""Image measurements shared by more than one feature.

Every function here is pure and stateless: it recomputes what it returns. Two
features may call the same measurement, but no value is ever cached or handed
from one feature to another — that independence is the point of one class per
feature, and it is what keeps a feature's cost attributable to that feature.

A measurement lives here only when more than one feature needs it; anything used
by a single feature belongs in that feature's module. Numeric *rules* (the
epsilon, the flat-input policy, the slope scaling) live in :mod:`_conventions`.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from scipy.signal import find_peaks

from ._conventions import EPS, safe_corr
from .options import FeatureOptions

LOCAL_STD_WINDOW = 5
COARSE_BLOCK = 16


class MassMoments(NamedTuple):
    """Intensity-weighted first and second moments, normalised to [0, 1]."""

    com_x: float
    com_y: float
    var_x: float
    var_y: float
    cov_xy: float


def peak_count(profile: np.ndarray, options: FeatureOptions) -> float:
    """Interior maxima with prominence of at least the configured fraction.

    The profile is padded with its own minimum on both ends so a maximum at the
    image border still counts; the padding can never create a false peak.
    """
    dynamic_range = float(profile.max() - profile.min())
    if dynamic_range < EPS:
        return 0.0
    padded = np.concatenate([[profile.min()], profile, [profile.min()]])
    peaks, _ = find_peaks(padded, prominence=dynamic_range * options.peak_prominence_frac)
    return float(len(peaks))


def roughness(profile: np.ndarray) -> float:
    """Total variation divided by the mean level: dimensionless lumpiness."""
    total_variation = float(np.abs(np.diff(profile)).sum())
    return total_variation / (float(profile.mean()) + EPS)


def diagonal_profile(image: np.ndarray, anti: bool) -> np.ndarray:
    """Mean intensity along every diagonal, from a dense projection.

    ``ndarray.diagonal`` is exact (no interpolation). ``anti`` reads the
    anti-diagonal by mirroring the columns first.
    """
    source = image[:, ::-1] if anti else image
    h, w = image.shape
    return np.asarray(
        [source.diagonal(offset=d).mean() for d in range(-(h - 1), w)],
        dtype=np.float64,
    )


def mass_moments(image: np.ndarray) -> MassMoments:
    """Intensity-weighted centre of mass and covariance, normalised to [0, 1].

    A black image has no mass: the centre is the middle of the frame and the
    variances are zero.
    """
    h, w = image.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    col_mass = image.sum(axis=0).astype(np.float64)
    row_mass = image.sum(axis=1).astype(np.float64)
    total = float(image.sum())
    if total < EPS:
        return MassMoments(0.5, 0.5, 0.0, 0.0, 0.0)

    com_x = float((col_mass * xs).sum() / total)
    com_y = float((row_mass * ys).sum() / total)
    var_x = float((col_mass * (xs - com_x) ** 2).sum() / total)
    var_y = float((row_mass * (ys - com_y) ** 2).sum() / total)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cov_xy = float((image.astype(np.float64) * (xx - com_x) * (yy - com_y)).sum() / total)
    return MassMoments(com_x, com_y, var_x, var_y, cov_xy)


def percentile_span(image: np.ndarray) -> float:
    """90th minus 10th intensity percentile: contrast without outlier sensitivity."""
    p10, p90 = np.percentile(image, 10), np.percentile(image, 90)
    return float(p90 - p10)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the preprocessed image."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def local_std_mean(image: np.ndarray) -> float:
    """Mean local standard deviation over 5x5 windows (E[x^2] - E[x]^2)."""
    image32 = image.astype(np.float32)
    mean = cv2.boxFilter(image32, -1, (LOCAL_STD_WINDOW,) * 2)
    mean_sq = cv2.boxFilter(image32 * image32, -1, (LOCAL_STD_WINDOW,) * 2)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return float(np.sqrt(var).mean())


def block_mean_variance(image: np.ndarray) -> float:
    """Variance of 16x16 block means (INTER_AREA downsampling)."""
    h, w = image.shape
    out_h, out_w = max(h // COARSE_BLOCK, 1), max(w // COARSE_BLOCK, 1)
    blocks = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return float(blocks.astype(np.float64).var())


def otsu_foreground(image: np.ndarray) -> np.ndarray:
    """Bright-foreground Otsu binarisation.

    cv2's THRESH_OTSU accepts only 8U/16U input, so the image is min-max scaled
    to uint8 and the threshold mapped back to the original scale (Otsu is
    invariant under affine rescaling, so this is exact). A flat image has no
    foreground. Polarity is fixed: brighter than threshold is foreground, so a
    genuinely dark-object-on-light image must be inverted by the caller.
    """
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < EPS:
        return np.zeros(image.shape, dtype=bool)
    u8 = np.round((image - lo) / (hi - lo) * 255.0).astype(np.uint8)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image > lo + float(threshold) / 255.0 * (hi - lo)


def mirror_corr(image: np.ndarray, axis: int) -> float:
    """Correlation of one half with the mirrored other half.

    ``axis=1`` splits on columns (left/right), ``axis=0`` on rows
    (top/bottom). The middle row/column is ignored when the size is odd. A
    single line is trivially symmetric with itself (1.0).
    """
    if axis == 1:
        half = image.shape[1] // 2
        a = image[:, :half]
        b = image[:, image.shape[1] - half :][:, ::-1]
    else:
        half = image.shape[0] // 2
        a = image[:half, :]
        b = image[image.shape[0] - half :, :][::-1, :]
    if half == 0:
        return 1.0
    return safe_corr(a, b, flat_policy="equal_is_one")


def colour_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(r, g, b) float64 planes, or None for grayscale input.

    Grayscale has no colour by definition, so colour features return 0.0 for
    it; the rule is stated once here rather than four times.
    """
    if image.ndim == 2:
        return None
    return (
        image[..., 0].astype(np.float64),
        image[..., 1].astype(np.float64),
        image[..., 2].astype(np.float64),
    )
