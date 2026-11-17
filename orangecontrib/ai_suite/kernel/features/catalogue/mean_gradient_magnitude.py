"""Overall roughness as the mean Sobel gradient magnitude."""

import numpy as np

from .._primitives import gradient_magnitude
from ..feature import Feature, FeatureOptions


class MeanGradientMagnitude(Feature):
    name = "mean_gradient_magnitude"
    definition = "Overall roughness: mean magnitude of the image gradient."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(gradient_magnitude(img).mean())
