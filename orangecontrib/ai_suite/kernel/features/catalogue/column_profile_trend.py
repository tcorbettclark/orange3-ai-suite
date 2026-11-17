"""Net intensity change across the image, from the dense column-mean profile."""

import numpy as np

from .._stats import scaled_slope
from ..feature import Feature, FeatureOptions


class ColumnProfileTrend(Feature):
    name = "column_profile_trend"
    definition = (
        "Net intensity change from image left to right (linear fit of the "
        "column-mean profile scaled by its length); positive = brighter on the right."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        columns = img.mean(axis=0)
        return scaled_slope(np.arange(len(columns), dtype=np.float64), columns)
