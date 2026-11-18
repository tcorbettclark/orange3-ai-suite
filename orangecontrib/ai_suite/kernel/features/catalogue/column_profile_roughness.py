import numpy as np

from .._imaging import roughness
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfileRoughness(Feature):
    name = "column_profile_roughness"
    description = Description(
        title="Horizontal bandiness",
        summary="Total variation of the column-mean brightness profile, divided by its mean.",
        intuition=(
            "How much the average brightness jumps up and down as you move across the "
            "picture. Vertical stripes, railings or a row of windows score high; a smooth "
            "left-to-right fade scores low. As with the vertical version, random speckle also "
            "scores high, so it is best read alongside the texture features."
        ),
        technical=(
            "Sum of absolute differences of the column-mean profile, divided by the profile "
            "mean: dimensionless. The same total-variation statistic as "
            "`row_profile_roughness`, applied to columns: vertical structure survives "
            "the averaging, horizontal structure does not."
        ),
        references=("[Total variation](https://en.wikipedia.org/wiki/Total_variation)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return roughness(image.mean(axis=0))
