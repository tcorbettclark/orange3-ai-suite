"""Bright bands along the anti-diagonal.

Kept as the pair to the main-diagonal count: the two disagree on asymmetric
structure, which is what makes them informative together.
"""

import numpy as np

from .._primitives import diagonal_profile, peak_count
from ..feature import Feature, FeatureOptions


class AntiDiagonalProfilePeaks(Feature):
    name = "anti_diagonal_profile_peaks"
    definition = (
        "Number of bright diagonal bands along the anti-diagonal, from a dense "
        "projection over all diagonals (no interpolation)."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return peak_count(diagonal_profile(img, anti=True), opts)
