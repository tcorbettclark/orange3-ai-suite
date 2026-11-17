"""How many vertical bands of bright material the image has."""

import numpy as np

from .._imaging import peak_count
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfilePeaks(Feature):
    name = "column_profile_peaks"
    definition = (
        "Number of vertical bands of bright material: maxima of the column-mean "
        "profile with prominence at least the configured fraction of its "
        "dynamic range; border maxima included."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return peak_count(img.mean(axis=0), opts)
