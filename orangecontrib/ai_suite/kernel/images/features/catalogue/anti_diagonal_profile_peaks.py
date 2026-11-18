import numpy as np

from .._measurements import diagonal_profile, peak_count
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class AntiDiagonalProfilePeaks(Feature):
    name = "anti_diagonal_profile_peaks"
    description = Description(
        title="Diagonal bands (anti)",
        summary="Number of bright bands along the anti-diagonal, from a dense projection over every diagonal.",
        intuition=(
            "Counts bright streaks running from the top-right to the bottom-left. A shadow "
            "falling one way across a wall scores here; a shadow falling the other way scores "
            "on the main-diagonal feature instead. A picture with no slanted structure scores "
            "none."
        ),
        technical=(
            "The columns are mirrored, then every diagonal is averaged with ndarray.diagonal "
            "and its peaks counted as in `row_profile_peaks`. Mirrored first because "
            "ndarray.diagonal offsets run top-left to bottom-right; the pairing with "
            "`diagonal_profile_peaks` is what makes asymmetry between the two "
            "diagonals visible."
        ),
        references=(
            "[NumPy diagonal](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return peak_count(diagonal_profile(image, anti=True), options)
