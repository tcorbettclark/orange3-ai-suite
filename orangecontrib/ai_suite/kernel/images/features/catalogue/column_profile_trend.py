import numpy as np

from .._conventions import scaled_slope
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfileTrend(Feature):
    name = "column_profile_trend"
    description = Description(
        title="Left-to-right trend",
        summary="Net change in brightness from the left of the image to the right.",
        intuition=(
            "Whether the picture gets brighter or darker as you look across. A room with its "
            "window on the right scores positive; the same room with the window on the left "
            "scores negative. A centred subject lit evenly on both sides scores near zero. "
            "Like its top-to-bottom twin, it only sees the average of each column."
        ),
        technical=(
            "The column-mean brightness profile is fitted with a least-squares line and the "
            "slope is multiplied by the number of columns. This is the net intensity change "
            "across the frame, positive towards the right. The transpose of "
            "`row_profile_trend`; the two disagree on images whose structure is not "
            "diagonal."
        ),
        references=(
            "[Simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)",
            "[NumPy polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        columns = image.mean(axis=0)
        return scaled_slope(np.arange(len(columns), dtype=np.float64), columns)
