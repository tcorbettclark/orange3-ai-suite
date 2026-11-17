"""Colourfulness: mean span between the brightest and dimmest channel."""

import numpy as np

from ..feature import Feature
from ..options import FeatureOptions


class MeanSaturation(Feature):
    name = "mean_saturation"
    definition = "Colourfulness: mean of (max - min) channel intensity."
    needs_colour = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        if img.ndim == 2:
            return 0.0
        return float((img.max(axis=-1) - img.min(axis=-1)).mean())
