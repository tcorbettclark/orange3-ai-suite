"""Vertical lumpiness: how much the row-mean profile wanders.

Total variation rather than variance, so a single sharp band counts as much as
a slow wave - which is the point: bands are structure, gradients are not.
"""

import numpy as np

from .._primitives import roughness
from ..feature import Feature, FeatureOptions


class RowProfileRoughness(Feature):
    name = "row_profile_roughness"
    definition = (
        "How lumpy the image is vertically: total variation of the row-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return roughness(img.mean(axis=1))
