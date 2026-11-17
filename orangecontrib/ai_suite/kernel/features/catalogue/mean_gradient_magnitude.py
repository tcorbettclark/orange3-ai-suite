"""Overall roughness as the mean Sobel gradient magnitude."""

import numpy as np

from .._imaging import gradient_magnitude
from ..feature import Feature
from ..options import FeatureOptions


class MeanGradientMagnitude(Feature):
    name = "mean_gradient_magnitude"
    definition = "Overall roughness: mean magnitude of the image gradient."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(gradient_magnitude(img).mean())
