import numpy as np

from .._imaging import peak_count
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class RowProfilePeaks(Feature):
    name = "row_profile_peaks"
    description = Description(
        title="Horizontal bands",
        summary="Number of distinct horizontal bands of brightness in the row-mean profile.",
        intuition=(
            "How many separate horizontal stripes of bright material you would see if you "
            "squinted. A sky with a low cloud layer and a high one scores two, a single band "
            "across the top scores one, and a smooth sky scores none. A band that runs off "
            "the edge of the picture still counts, because the cut edge is not treated as a "
            "peak."
        ),
        technical=(
            "Local maxima of the row-mean profile found with scipy.signal.find_peaks, with "
            "prominence at least options.peak_prominence_frac times the profile's dynamic "
            "range (default 0.10). The profile is padded with its own minimum at both ends so "
            "a maximum on the border counts, and padding cannot create a peak. A whole "
            "number, and exactly stable under small shifts when the bands are unambiguous."
        ),
        references=(
            "[scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return peak_count(image.mean(axis=1), options)
