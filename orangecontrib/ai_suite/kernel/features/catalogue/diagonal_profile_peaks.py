import numpy as np

from .._imaging import diagonal_profile, peak_count
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class DiagonalProfilePeaks(Feature):
    name = "diagonal_profile_peaks"
    description = Description(
        title="Diagonal bands (main)",
        summary="Number of bright bands along the main diagonal, from a dense projection over every diagonal.",
        intuition=(
            "Counts bright streaks running from the top-left to the bottom-right, as you "
            "would see in a contrail or the grain of brushed wood. One clean diagonal streak "
            "scores one; two parallel streaks score two; a picture with no diagonal structure "
            "scores none. It is the diagonal counterpart of the horizontal and vertical band "
            "counts."
        ),
        technical=(
            "Every diagonal of the image is averaged with ndarray.diagonal, giving a dense "
            "projection with no interpolation or sampling; peaks are then counted as in "
            "`row_profile_peaks`. An anti-diagonal streak appears in "
            "`anti_diagonal_profile_peaks` instead."
        ),
        references=(
            "[NumPy diagonal](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return peak_count(diagonal_profile(image, anti=False), options)
