"""Which way the intensity distribution leans.

Positive skew is a small bright object on a large dark field - the usual
picture of an isolated cloud, or of ink on paper.
"""

import numpy as np

from .._conventions import is_flat
from ..feature import Feature
from ..options import FeatureOptions


class Skewness(Feature):
    name = "skewness"
    definition = (
        "Bright blobs on dark background (>0) vs. dark blobs on bright (<0); "
        "0 when the intensity distribution is symmetric or flat."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        mean, std = float(image.mean()), float(image.std())
        z = image - mean
        return float((z**3).mean() / std**3)
