"""Excess kurtosis: peaked, flat-topped, or Gaussian-like.

Computed as E[z^4] - 3 so the reference point is the Gaussian, not the
uniform distribution.
"""

import numpy as np

from .._conventions import is_flat
from ..feature import Feature
from ..options import FeatureOptions


class Kurtosis(Feature):
    name = "kurtosis"
    definition = "Intensity-distribution shape: >0 peaky, <0 flat-topped, ~0 Gaussian-like."

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        mean, std = float(image.mean()), float(image.std())
        z = image - mean
        return float((z**4).mean() / std**4 - 3.0)
