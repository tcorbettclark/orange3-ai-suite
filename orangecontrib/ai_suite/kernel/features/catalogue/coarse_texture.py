"""Big lumps: variance of the 16x16 block means."""

import numpy as np

from .._imaging import block_mean_variance
from ..feature import Feature
from ..options import FeatureOptions


class CoarseTexture(Feature):
    name = "coarse_texture"
    definition = "Big lumps vs. uniform: variance of the 16x16 block means."

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return block_mean_variance(image)
