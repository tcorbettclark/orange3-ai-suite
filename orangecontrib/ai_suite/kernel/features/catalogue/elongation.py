"""Streak or blob, from the eigenvalues of the mass covariance matrix.

The ratio is orientation-free, so a diagonal streak scores like a horizontal
one. A degenerate (single-line) covariance returns 1.0 rather than infinity.
"""

import numpy as np

from .._conventions import EPS
from .._imaging import mass_moments
from ..feature import Feature
from ..options import FeatureOptions


class Elongation(Feature):
    name = "elongation"
    definition = (
        "Streak (>1) vs. blob (~1): eigenvalue ratio of the intensity second-moment matrix."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        moments = mass_moments(image)
        matrix = np.array([[moments.var_x, moments.cov_xy], [moments.cov_xy, moments.var_y]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        smallest = float(eigenvalues[0])
        return float(eigenvalues[1] / smallest) if smallest > EPS else 1.0
