import numpy as np

from .._conventions import EPS
from .._measurements import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class Elongation(Feature):
    name = "elongation"
    description = Description(
        title="Elongation",
        summary="Ratio of the larger to the smaller eigenvalue of the intensity covariance: streak versus blob.",
        intuition=(
            "Streak or lump? A contrail, a horizon or a stretched shadow has one long "
            "direction and one short one, so the ratio is large. A ball, a round cloud or a "
            "centred face is about as wide as it is tall, so the ratio sits near 1. The "
            "direction does not matter: a diagonal streak scores the same as a horizontal "
            "one."
        ),
        technical=(
            "Eigenvalue ratio of the 2x2 intensity covariance matrix of the mass "
            "distribution, computed with numpy.linalg.eigvalsh. Values are >= 1; a degenerate "
            "covariance (all mass on a line, or an image too small to have spread) returns "
            "1.0. Orientation-free by construction, which is why it is not accompanied by a "
            "separate angle feature."
        ),
        references=("[Image moment](https://en.wikipedia.org/wiki/Image_moment)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        moments = mass_moments(image)
        matrix = np.array([[moments.var_x, moments.cov_xy], [moments.cov_xy, moments.var_y]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        smallest = float(eigenvalues[0])
        return float(eigenvalues[1] / smallest) if smallest > EPS else 1.0
