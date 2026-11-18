import numpy as np

from .._imaging import mirror_corr
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class LeftRightSymmetry(Feature):
    name = "left_right_symmetry"
    description = Description(
        title="Mirror symmetry (left/right)",
        summary="Correlation between the left half of the image and the mirrored right half.",
        intuition=(
            "How closely the left side mirrors the right. A face looking straight at the "
            "camera, a centred building or a reflection in still water scores high. A "
            "shoreline running diagonally, or a subject standing to one side, scores low. It "
            "compares the two halves pixel by pixel, so a small shift of the subject breaks "
            "the match even if the scene still looks balanced."
        ),
        technical=(
            "Pearson coefficient between the left half and the reversed right half; the "
            "middle column is dropped when the width is odd. A flat pair returns 1.0 when "
            "both halves are constant and equal, 0.0 when one is flat or both are constant "
            "but unequal, so uniform skies cannot produce a NaN. See "
            "`top_bottom_symmetry` and `quadrant_self_similarity`."
        ),
        references=(
            "[Reflection symmetry](https://en.wikipedia.org/wiki/Reflection_symmetry)",
            "[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return mirror_corr(image, axis=1)
