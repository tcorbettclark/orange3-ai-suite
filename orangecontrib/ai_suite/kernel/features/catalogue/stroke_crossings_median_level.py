"""Dense version of the classic OCR stroke-crossing count.

Counted per row against the image's *global* median level, then averaged over
rows - unlike ``profile_median_crossing_count``, which uses each line's own
median and averages rows with columns. The two are near-cousins by name only:
this one measures how many strokes a horizontal scan passes through, which is
why open digits ("1", "7") score low and dense ones ("8") score high.

Below-or-equal counts as below, which keeps the count stable when the median
coincides with the background level of a minority bright object.
"""

import numpy as np

from ..feature import Feature, FeatureOptions


class StrokeCrossingsMedianLevel(Feature):
    name = "stroke_crossings_median_level"
    definition = (
        "Dense version of the classic OCR stroke-crossing count: crossings of "
        "the global median level per row, averaged; open strokes ('1', '7') "
        "score low, dense strokes ('8') high."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        below = img <= np.median(img)
        transitions = (below[:, 1:] != below[:, :-1]).sum(axis=1)
        return float(transitions.mean())
