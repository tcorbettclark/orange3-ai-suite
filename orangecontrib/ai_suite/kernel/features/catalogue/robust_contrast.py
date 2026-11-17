"""Contrast that one bright speck cannot move: a percentile span.

Kept alongside ``intensity_std`` because the pair is diagnostic: a large std
with a small percentile span means the image carries a few extreme pixels
rather than broad tonal range.
"""

import numpy as np

from ..feature import Feature, FeatureOptions


class RobustContrast(Feature):
    name = "robust_contrast"
    definition = "Contrast without outlier sensitivity: 90th minus 10th intensity percentile."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        p10, p90 = np.percentile(img, 10), np.percentile(img, 90)
        return float(p90 - p10)
