"""Group A: global photometry (6 features).

Overall brightness and contrast character of the image. All operate on the
preprocessed [0, 1] luminance image (see preprocess.py).
"""

from __future__ import annotations

import numpy as np

from .options import FeatureOptions

_EPS = 1e-12

FEATURE_NAMES = (
    "mean_intensity",
    "intensity_std",
    "robust_contrast",
    "skewness",
    "kurtosis",
    "bright_fraction",
)

DEFINITIONS = {
    "mean_intensity": "Overall brightness: mean pixel intensity.",
    "intensity_std": "Contrast: standard deviation of pixel intensity.",
    "robust_contrast": (
        "Contrast without outlier sensitivity: 90th minus 10th intensity percentile."
    ),
    "skewness": (
        "Bright blobs on dark background (>0) vs. dark blobs on bright (<0); "
        "0 when the intensity distribution is symmetric or flat."
    ),
    "kurtosis": ("Intensity-distribution shape: >0 peaky, <0 flat-topped, ~0 Gaussian-like."),
    "bright_fraction": (
        "Fraction of pixels brighter than the image's own median + "
        "robust_contrast/2 — adaptive 'how much bright stuff'."
    ),
}


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    p10, p90 = np.percentile(img, 10), np.percentile(img, 90)
    robust = float(p90 - p10)
    mean = float(img.mean())
    std = float(img.std())

    if std < _EPS:
        skew, kurt = 0.0, 0.0
    else:
        z = img - mean
        skew = float((z**3).mean() / std**3)
        # Excess kurtosis: 0 = Gaussian-like, >0 peaky, <0 flat-topped.
        kurt = float((z**4).mean() / std**4 - 3.0)

    threshold = float(np.median(img)) + robust / 2.0
    bright = float((img > threshold).mean())

    return {
        "mean_intensity": mean,
        "intensity_std": std,
        "robust_contrast": robust,
        "skewness": skew,
        "kurtosis": kurt,
        "bright_fraction": bright,
    }
