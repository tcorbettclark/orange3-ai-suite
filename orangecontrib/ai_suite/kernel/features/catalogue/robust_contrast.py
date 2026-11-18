"""Contrast that one bright speck cannot move: a percentile span.

Kept alongside ``intensity_std`` because the pair is diagnostic: a large std
with a small percentile span means the image carries a few extreme pixels
rather than broad tonal range.
"""

import numpy as np

from .._imaging import percentile_span
from ..feature import Feature
from ..options import FeatureOptions


class RobustContrast(Feature):
    name = "robust_contrast"
    definition = "Contrast without outlier sensitivity: 90th minus 10th intensity percentile."

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return percentile_span(image)
