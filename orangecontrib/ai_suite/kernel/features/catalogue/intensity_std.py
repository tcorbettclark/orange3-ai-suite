"""Contrast as plain dispersion, before the outlier-robust variant."""

import numpy as np

from ..feature import Feature
from ..options import FeatureOptions


class IntensityStd(Feature):
    name = "intensity_std"
    definition = "Contrast: standard deviation of pixel intensity."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(img.std())
