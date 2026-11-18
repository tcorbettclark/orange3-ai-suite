"""Vertical lumpiness: how much the row-mean profile wanders.

Total variation rather than variance, so a single sharp band counts as much as
a slow wave - which is the point: bands are structure, gradients are not.
"""

import numpy as np

from .._imaging import roughness
from ..feature import Feature
from ..options import FeatureOptions


class RowProfileRoughness(Feature):
    name = "row_profile_roughness"
    definition = (
        "How lumpy the image is vertically: total variation of the row-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return roughness(image.mean(axis=1))
