"""Streak or blob, from the eigenvalues of the mass covariance matrix.

The ratio is orientation-free, so a diagonal streak scores like a horizontal
one. A degenerate (single-line) covariance returns 1.0 rather than infinity.
"""

import numpy as np

from .._primitives import mass_moments
from .._stats import EPS
from ..feature import Feature, FeatureOptions


class Elongation(Feature):
    name = "elongation"
    definition = (
        "Streak (>1) vs. blob (~1): eigenvalue ratio of the intensity second-moment matrix."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        moments = mass_moments(img)
        matrix = np.array([[moments.var_x, moments.cov_xy], [moments.cov_xy, moments.var_y]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        smallest = float(eigenvalues[0])
        return float(eigenvalues[1] / smallest) if smallest > EPS else 1.0
