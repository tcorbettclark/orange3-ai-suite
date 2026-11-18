import numpy as np

from .._measurements import mirror_corr
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class TopBottomSymmetry(Feature):
    name = "top_bottom_symmetry"
    description = Description(
        title="Mirror symmetry (top/bottom)",
        summary="Correlation between the top half of the image and the mirrored bottom half.",
        intuition=(
            "How closely the top mirrors the bottom. A reflection in a lake, a ceiling above "
            "an identical floor, or a centred ball on a mirror table scores high. A landscape "
            "with sky above and ground below scores low, which is what makes this feature "
            "useful: most ordinary photographs are not symmetric this way."
        ),
        technical=(
            "Pearson coefficient between the top half and the reversed bottom half; the "
            "middle row is dropped when the height is odd. The same flat-pair rule as "
            "`left_right_symmetry`: equal constant halves score 1.0, otherwise 0.0. "
            "Photographs of clouds often score higher here than left/right, because skies are "
            "layered rather than mirrored sideways."
        ),
        references=(
            "[Reflection symmetry](https://en.wikipedia.org/wiki/Reflection_symmetry)",
            "[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return mirror_corr(image, axis=0)
