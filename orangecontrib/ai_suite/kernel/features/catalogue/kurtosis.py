"""Excess kurtosis: peaked, flat-topped, or Gaussian-like.

Computed as E[z^4] - 3 so the reference point is the Gaussian, not the
uniform distribution.
"""

import numpy as np

from .._conventions import EPS
from ..feature import Feature
from ..options import FeatureOptions


class Kurtosis(Feature):
    name = "kurtosis"
    definition = "Intensity-distribution shape: >0 peaky, <0 flat-topped, ~0 Gaussian-like."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        mean, std = float(img.mean()), float(img.std())
        if std < EPS:
            return 0.0
        z = img - mean
        return float((z**4).mean() / std**4 - 3.0)
