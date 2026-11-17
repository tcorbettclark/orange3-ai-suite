"""Bright bands along the main diagonal.

The projection is over every diagonal, so nothing is sampled or interpolated;
a band that straddles two diagonals still reads as one maximum because the
profile is continuous.
"""

import numpy as np

from .._imaging import diagonal_profile, peak_count
from ..feature import Feature
from ..options import FeatureOptions


class DiagonalProfilePeaks(Feature):
    name = "diagonal_profile_peaks"
    definition = (
        "Number of bright diagonal bands along the main diagonal, from a dense "
        "projection over all diagonals (no interpolation)."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return peak_count(diagonal_profile(img, anti=False), opts)
