import numpy as np

from .._conventions import safe_corr
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class QuadrantSelfSimilarity(Feature):
    name = "quadrant_self_similarity"
    description = Description(
        title="Quadrant similarity",
        summary="Mean correlation among the six pairs of image quadrants.",
        intuition=(
            "Do the four corners of the picture look like each other? Foliage, gravel or a "
            "wall filling the frame scores high, because every corner is much like the next. "
            "A portrait, where one corner holds the subject and the rest holds background, "
            "scores low. A single object with an even surround lands somewhere in between."
        ),
        technical=(
            "The image is split into four quadrants and the six pairwise Pearson coefficients "
            "are averaged. An odd row or column is dropped rather than resampled, consistent "
            "with `left_right_symmetry`. Constant quadrants follow the same flat rule "
            "as the mirror features. High for texture that is uniform across the frame, low "
            "for one localised object."
        ),
        references=(
            "[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)",
        ),
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
