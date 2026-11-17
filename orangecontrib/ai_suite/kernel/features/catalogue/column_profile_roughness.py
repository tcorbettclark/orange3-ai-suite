"""Horizontal lumpiness: how much the column-mean profile wanders."""

import numpy as np

from .._primitives import roughness
from ..feature import Feature, FeatureOptions


class ColumnProfileRoughness(Feature):
    name = "column_profile_roughness"
    definition = (
        "How lumpy the image is horizontally: total variation of the column-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return roughness(img.mean(axis=0))
