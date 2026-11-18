import numpy as np

from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class StrokeCrossingCount(Feature):
    name = "stroke_crossing_count"
    description = Description(
        title="Stroke crossings",
        summary="Mean number of times a row crosses the global median brightness.",
        intuition=(
            "Imagine running a finger along each row, counting every time you move from the "
            "darker half of the picture to the brighter half or back. Dense lettering scores "
            "high, a single wide band scores low, and a smooth sky scores none. Rows only: "
            "this measures how many strokes a horizontal scan passes through, which is the "
            "classic way to tell a dense 8 from an open 1."
        ),
        technical=(
            "Each row is compared with the image's global median and the number of "
            "transitions between below and above is counted; the mean over rows is reported. "
            "Real-valued because it averages. Below-or-equal counts as below, which keeps the "
            "count stable when the median coincides with the background of a minority bright "
            "object. Rows only, and one global level: `profile_median_crossing_count` uses "
            "each line's own median, both axes, and counts upward crossings only, where this "
            "counts transitions in both directions."
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        below = image <= np.median(image)
        transitions = (below[:, 1:] != below[:, :-1]).sum(axis=1)
        return float(transitions.mean())
