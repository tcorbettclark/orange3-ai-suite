import numpy as np

from .._conventions import scaled_slope
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class RowProfileTrend(Feature):
    name = "row_profile_trend"
    description = Description(
        title="Top-to-bottom trend",
        summary="Net change in brightness from the top of the image to the bottom.",
        intuition=(
            "Whether the picture gets brighter or darker as you look down. A sunset with a "
            "bright horizon over dark ground scores positive; a bright ceiling above a dark "
            "floor scores negative. A scene that looks much the same all the way down scores "
            "near zero. It only sees the average of each row, so left-right detail does not "
            "matter."
        ),
        technical=(
            "The row-mean brightness profile is fitted with a least-squares line and the "
            "slope is multiplied by the number of rows. The result is the net change in "
            "intensity units across the frame (roughly [-1, 1] after normalisation), not a "
            "per-row rate. Positive means brighter at the bottom. The transpose of "
            "`column_profile_trend`."
        ),
        references=(
            "[Simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)",
            "[NumPy polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        rows = image.mean(axis=1)
        return scaled_slope(np.arange(len(rows), dtype=np.float64), rows)
