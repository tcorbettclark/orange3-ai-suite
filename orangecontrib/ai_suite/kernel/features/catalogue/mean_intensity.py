"""Brightness means brightness *after* contrast normalisation.

The pipeline rescales every image to its own [p1, p99] range, so this is a
relative level, not an exposure. That is what makes it comparable across
images, and also why it cannot answer "was this photo dark?".
"""

import numpy as np

from ..feature import Feature
from ..options import FeatureOptions


class MeanIntensity(Feature):
    name = "mean_intensity"
    definition = "Overall brightness: mean pixel intensity."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(img.mean())
