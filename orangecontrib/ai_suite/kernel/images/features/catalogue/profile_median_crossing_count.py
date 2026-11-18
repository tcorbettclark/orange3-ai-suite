"""The slowest feature in the suite; the first target if extraction needs speed."""

import numpy as np

from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


def _upcrossings(line: np.ndarray) -> float:
    above = line > np.median(line)
    return float((above[1:] & ~above[:-1]).sum())


class ProfileMedianCrossingCount(Feature):
    name = "profile_median_crossing_count"
    description = Description(
        title="Median crossings",
        summary="Median number of times a row or column crosses its own median brightness.",
        intuition=(
            "Imagine scanning along each row and each column and counting how often you move "
            "from the darker half to the brighter half. Vertical stripes crossed left to "
            "right give several; a smooth sky gives none. Because the level is each line's "
            "own middle value, dim and bright pictures are counted the same way, with nothing "
            "to tune."
        ),
        technical=(
            "For each row, then each column, the line's own median is found and the number of "
            "upward crossings counted; the median over lines is taken, and rows and columns "
            "are averaged. Real-valued because it averages over lines. The level is per line, "
            "unlike `stroke_crossing_count`, which uses one global level and rows "
            "only."
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        rows = [_upcrossings(image[i, :]) for i in range(image.shape[0])]
        columns = [_upcrossings(image[:, j]) for j in range(image.shape[1])]
        return float((float(np.median(rows)) + float(np.median(columns))) / 2.0)
