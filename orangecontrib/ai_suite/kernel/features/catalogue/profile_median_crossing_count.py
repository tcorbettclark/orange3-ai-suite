"""Level crossings, with the level taken from each line itself.

The threshold is each row's and column's own median, so the count is
parameter-free: no absolute level to calibrate, no sensitivity knob. Rows and
columns are counted separately and averaged, because a vertical bar crosses
every row once but leaves the columns flat.

This is the slowest feature in the suite (it loops over 2 * size lines, each in
numpy), and the obvious target if extraction ever needs to be faster.
"""

import numpy as np

from ..feature import Feature, FeatureOptions


def _upcrossings(line: np.ndarray) -> float:
    above = line > np.median(line)
    return float((above[1:] & ~above[:-1]).sum())


class ProfileMedianCrossingCount(Feature):
    name = "profile_median_crossing_count"
    definition = (
        "How often pixel lines cross their own median level — parameter-free "
        "cousin of peak counting (median over rows and columns, averaged)."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        rows = [_upcrossings(img[i, :]) for i in range(img.shape[0])]
        columns = [_upcrossings(img[:, j]) for j in range(img.shape[1])]
        return float((float(np.median(rows)) + float(np.median(columns))) / 2.0)
