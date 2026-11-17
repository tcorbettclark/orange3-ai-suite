"""Big lumps: variance of the 16x16 block means."""

import numpy as np

from .._primitives import block_mean_variance
from ..feature import Feature, FeatureOptions


class CoarseTexture(Feature):
    name = "coarse_texture"
    definition = "Big lumps vs. uniform: variance of the 16x16 block means."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return block_mean_variance(img)
