"""Level crossings, with the level taken from each line itself.

The threshold is each row's and column's own median, so the count is
parameter-free: no absolute level to calibrate, no sensitivity knob. Rows and
columns are counted separately and averaged, because a vertical bar crosses
every row once but leaves the columns flat.

This is the slowest feature in the suite (it loops over 2 * size lines, each in
numpy), and the obvious target if extraction ever needs to be faster.
"""

import numpy as np

from ..feature import Feature
from ..options import FeatureOptions


def _upcrossings(line: np.ndarray) -> float:
    above = line > np.median(line)
    return float((above[1:] & ~above[:-1]).sum())


class ProfileMedianCrossingCount(Feature):
    name = "profile_median_crossing_count"
    definition = (
        "How often pixel lines cross their own median level — parameter-free "
        "cousin of peak counting (median over rows and columns, averaged)."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        rows = [_upcrossings(image[i, :]) for i in range(image.shape[0])]
        columns = [_upcrossings(image[:, j]) for j in range(image.shape[1])]
        return float((float(np.median(rows)) + float(np.median(columns))) / 2.0)
