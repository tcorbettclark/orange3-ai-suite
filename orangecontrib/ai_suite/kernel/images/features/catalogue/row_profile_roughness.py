import numpy as np

from .._measurements import roughness
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class RowProfileRoughness(Feature):
    name = "row_profile_roughness"
    description = Description(
        title="Horizontal band strength",
        summary="Total variation of the row-mean brightness profile, divided by its mean.",
        intuition=(
            "How much the average brightness jumps up and down as you move down the picture. "
            "A sky with two distinct cloud layers, or a striped shirt, scores high; a smooth "
            "gradient from top to bottom scores low. Random speckle also scores high, so on "
            "its own it does not say whether the bands are organised or just noisy."
        ),
        technical=(
            "Sum of absolute differences of the row-mean profile, divided by the profile "
            "mean: dimensionless. Total variation counts every change of direction equally, "
            "so several thin bands score higher than one wide band of the same overall "
            "contrast. A property of the projection, not of single pixels: horizontal "
            "structure survives averaging, vertical structure averages away."
        ),
        references=("[Total variation](https://en.wikipedia.org/wiki/Total_variation)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return roughness(image.mean(axis=1))
