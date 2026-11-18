"""Colourfulness: mean span between the brightest and dimmest channel."""

import numpy as np

from .._imaging import colour_channels
from ..feature import Feature
from ..options import FeatureOptions


class MeanSaturation(Feature):
    name = "mean_saturation"
    definition = "Colourfulness: mean of (max - min) channel intensity."
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        planes = np.stack(channels)
        return float((planes.max(axis=0) - planes.min(axis=0)).mean())
