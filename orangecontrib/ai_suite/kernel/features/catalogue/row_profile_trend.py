"""Net intensity change down the image, from the dense row-mean profile.

The slope is scaled by the profile length, so the value is a change in
intensity units rather than a per-row rate that shrinks as images grow.
"""

import numpy as np

from .._conventions import scaled_slope
from ..feature import Feature
from ..options import FeatureOptions


class RowProfileTrend(Feature):
    name = "row_profile_trend"
    definition = (
        "Net intensity change from image top to bottom (linear fit of the "
        "row-mean profile scaled by its length); positive = brighter at the bottom."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        rows = img.mean(axis=1)
        return scaled_slope(np.arange(len(rows), dtype=np.float64), rows)
