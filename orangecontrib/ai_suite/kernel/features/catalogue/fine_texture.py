"""Small-scale lumpiness: mean local standard deviation over 5x5 windows."""

import numpy as np

from .._imaging import local_std_mean
from ..feature import Feature
from ..options import FeatureOptions


class FineTexture(Feature):
    name = "fine_texture"
    definition = "Small-scale lumpiness: mean local standard deviation over 5x5 windows."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return local_std_mean(img)
