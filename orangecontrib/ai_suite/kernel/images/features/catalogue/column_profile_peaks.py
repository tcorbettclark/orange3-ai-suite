import numpy as np

from .._measurements import peak_count
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfilePeaks(Feature):
    name = "column_profile_peaks"
    description = Description(
        title="Vertical bands",
        summary="Number of distinct vertical bands of brightness in the column-mean profile.",
        intuition=(
            "How many separate upright stripes of bright material you would count across the "
            "picture. Two pillars of light score two; a single bright doorway scores one; an "
            "even sky scores none. A stripe that touches the left or right edge still counts, "
            "because the cut edge is not treated as a peak."
        ),
        technical=(
            "Local maxima of the column-mean profile found with scipy.signal.find_peaks, with "
            "prominence at least options.peak_prominence_frac times the profile's dynamic "
            "range (default 0.10). The profile is padded with its own minimum at both ends so "
            "an edge maximum counts. The transpose of `row_profile_peaks`, and a "
            "whole number."
        ),
        references=(
            "[scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return peak_count(image.mean(axis=0), options)
