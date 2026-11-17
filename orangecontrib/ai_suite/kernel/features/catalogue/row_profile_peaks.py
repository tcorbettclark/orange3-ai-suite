"""How many horizontal bands of bright material the image has.

Peaks are counted on the row-mean profile, not on the image, so a band must
span the width to register. Prominence is relative to the profile's own
dynamic range, and the profile is padded with its minimum so a band touching
the top or bottom border still counts.
"""

import numpy as np

from .._primitives import peak_count
from ..feature import Feature, FeatureOptions


class RowProfilePeaks(Feature):
    name = "row_profile_peaks"
    definition = (
        "Number of horizontal bands of bright material: maxima of the row-mean "
        "profile with prominence at least the configured fraction of its "
        "dynamic range; border maxima included."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return peak_count(img.mean(axis=1), opts)
