"""How much bright stuff, with the threshold taken from the image itself.

The threshold sits half a robust contrast above the image's own median, so a
dim image and a bright image are judged by the same relative standard. A fixed
absolute threshold would silently encode the dataset's lighting.
"""

import numpy as np

from .._imaging import percentile_span
from ..feature import Feature
from ..options import FeatureOptions


class BrightFraction(Feature):
    name = "bright_fraction"
    definition = (
        "Fraction of pixels brighter than the image's own median + "
        "robust_contrast/2 — adaptive 'how much bright stuff'."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        threshold = float(np.median(image)) + percentile_span(image) / 2.0
        return float((image > threshold).mean())
