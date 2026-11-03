"""Group D — texture / smoothness (6 features): the clouds workhorse.

Definitions
-----------
- mean gradient magnitude: mean of the Sobel gradient magnitude.
- edge density: fraction of pixels whose gradient magnitude reaches
  ``edge_density_frac`` (default 0.5) x the image's own gradient p99.
  Adaptive but non-degenerate: a fixed percentile threshold would return a
  constant 25% for every image (documented in features/README.md).
- fine texture: mean local standard deviation over 5x5 windows, via
  E[x^2] - E[x]^2 (box filter).
- coarse texture: variance of 16x16 block means (INTER_AREA downsampling).
- coarse-to-fine ratio: coarse texture as a fraction of total texture,
  ``coarse / (coarse + fine + eps)`` in [0, 1] (0.5 = scale-neutral). The
  spec's ``coarse / fine`` divides two small numbers and is ill-conditioned
  exactly when images are smooth (documented deviation 5 in README.md);
  puffy cumulus scores high, wispy cirrus low, identically ranked.
- autocorrelation half-width: typical feature size in pixels. The 2-D
  autocorrelation (FFT-based, mean-subtracted) is normalised by its central
  value, radially averaged, and the first radius where the radial profile
  drops below 0.5 is returned (linearly interpolated). If it never drops,
  the maximum radius is returned. A flat (mean-subtracted zero) image
  returns 0.0.
"""

from __future__ import annotations

import cv2
import numpy as np

from .options import FeatureOptions

_EPS = 1e-12
_BLOCK = 16
_LOCAL_STD_WINDOW = 5

FEATURE_NAMES = (
    "mean_gradient_magnitude",
    "edge_density",
    "fine_texture",
    "coarse_texture",
    "coarse_to_fine_texture_ratio",
    "autocorrelation_half_width",
)

DEFINITIONS = {
    "mean_gradient_magnitude": (
        "Overall roughness: mean magnitude of the image gradient."
    ),
    "edge_density": (
        "Fraction of 'edgy' pixels: gradient at least half the image's own "
        "p99 gradient (adaptive)."
    ),
    "fine_texture": (
        "Small-scale lumpiness: mean local standard deviation over 5x5 windows."
    ),
    "coarse_texture": (
        "Big lumps vs. uniform: variance of the 16x16 block means."
    ),
    "coarse_to_fine_texture_ratio": (
        "Where the texture lives: coarse texture as a fraction of total "
        "texture (coarse/(coarse+fine), [0,1], 0.5 = scale-neutral); puffy "
        "cumulus scores high, wispy cirrus low."
    ),
    "autocorrelation_half_width": (
        "Typical feature size in pixels: first lag where the normalised "
        "autocorrelation drops below 0.5."
    ),
}


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_density(mag: np.ndarray, opts: FeatureOptions) -> float:
    p99 = float(np.percentile(mag, 99))
    if p99 < _EPS:
        return 0.0
    return float((mag >= opts.edge_density_frac * p99).mean())


def _local_std_mean(img: np.ndarray) -> float:
    img32 = img.astype(np.float32)
    mean = cv2.boxFilter(img32, -1, (_LOCAL_STD_WINDOW,) * 2)
    mean_sq = cv2.boxFilter(img32 * img32, -1, (_LOCAL_STD_WINDOW,) * 2)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return float(np.sqrt(var).mean())


def _block_mean_variance(img: np.ndarray) -> float:
    h, w = img.shape
    out_h, out_w = max(h // _BLOCK, 1), max(w // _BLOCK, 1)
    blocks = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return float(blocks.astype(np.float64).var())


def _autocorr_half_width(img: np.ndarray) -> float:
    h, w = img.shape
    x = img.astype(np.float64) - img.mean()
    if x.std() < _EPS:
        return 0.0
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2).real)
    ac /= ac[h // 2, w // 2]  # normalise by lag-0 (the variance)
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2).round().astype(int)
    sums = np.bincount(r.ravel(), weights=ac.ravel())
    counts = np.bincount(r.ravel())
    prof = sums / np.maximum(counts, 1)
    rmax = len(prof) - 1
    for radius in range(1, rmax + 1):
        if prof[radius] < 0.5:
            prev, cur = prof[radius - 1], prof[radius]
            frac = 0.0 if cur >= prev else (prev - 0.5) / (prev - cur)
            return float(radius - 1 + frac)
    return float(rmax)


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    mag = _gradient_magnitude(img)
    fine = _local_std_mean(img)
    coarse = _block_mean_variance(img)
    return {
        "mean_gradient_magnitude": float(mag.mean()),
        "edge_density": _edge_density(mag, opts),
        "fine_texture": fine,
        "coarse_texture": coarse,
        "coarse_to_fine_texture_ratio": coarse / (coarse + fine + _EPS),
        "autocorrelation_half_width": _autocorr_half_width(img),
    }