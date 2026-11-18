"""Horizontal lumpiness: how much the column-mean profile wanders."""

import numpy as np

from .._imaging import roughness
from ..feature import Feature
from ..options import FeatureOptions


class ColumnProfileRoughness(Feature):
    name = "column_profile_roughness"
    definition = (
        "How lumpy the image is horizontally: total variation of the column-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return roughness(image.mean(axis=0))
