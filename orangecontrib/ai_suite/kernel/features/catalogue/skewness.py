"""Which way the intensity distribution leans.

Positive skew is a small bright object on a large dark field - the usual
picture of an isolated cloud, or of ink on paper.
"""

import numpy as np

from .._conventions import EPS
from ..feature import Feature
from ..options import FeatureOptions


class Skewness(Feature):
    name = "skewness"
    definition = (
        "Bright blobs on dark background (>0) vs. dark blobs on bright (<0); "
        "0 when the intensity distribution is symmetric or flat."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        mean, std = float(img.mean()), float(img.std())
        if std < EPS:
            return 0.0
        z = img - mean
        return float((z**3).mean() / std**3)
