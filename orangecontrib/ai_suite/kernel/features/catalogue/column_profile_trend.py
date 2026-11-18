"""Net intensity change across the image, from the dense column-mean profile."""

import numpy as np

from .._conventions import scaled_slope
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfileTrend(Feature):
    name = "column_profile_trend"
    definition = (
        "Net intensity change from image left to right (linear fit of the "
        "column-mean profile scaled by its length); positive = brighter on the right."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        columns = image.mean(axis=0)
        return scaled_slope(np.arange(len(columns), dtype=np.float64), columns)
