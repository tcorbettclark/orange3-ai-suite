"""Do the four quadrants look like each other?

The mean of the six pairwise inter-quadrant correlations: high for uniform
structure spread everywhere, low for one localised object. An odd size means
the four quadrants would differ by a pixel, so the last row or column is
dropped - the same rule :func:`._imaging.mirror_corr` uses for half symmetry.
"""

import numpy as np

from .._conventions import safe_corr
from ..feature import Feature
from ..options import FeatureOptions


class QuadrantSelfSimilarity(Feature):
    name = "quadrant_self_similarity"
    definition = (
        "Uniform vs. localised structure: mean of the six pairwise inter-quadrant correlations."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        h, w = image.shape
        if h < 2 or w < 2:
            return 0.0
        h2, w2 = h // 2, w // 2
        core = image[: 2 * h2, : 2 * w2]
        quadrants = [core[:h2, :w2], core[:h2, w2:], core[h2:, :w2], core[h2:, w2:]]
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        correlations = [
            safe_corr(quadrants[i], quadrants[j], flat_policy="equal_is_one") for i, j in pairs
        ]
        return float(np.mean(correlations))
