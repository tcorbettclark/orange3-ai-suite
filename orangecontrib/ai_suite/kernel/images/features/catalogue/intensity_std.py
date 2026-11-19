import numpy as np

from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class IntensityStd(Feature):
    name = "intensity_std"
    description = Description(
        title="Intensity spread",
        summary="Standard deviation of brightness across the preprocessed image.",
        intuition=(
            "How far pixels sit from the average brightness. A chessboard, or black text on "
            "white paper, scores high because most pixels are at one extreme or the other. "
            "Grey fog, a plain wall or an empty sky scores low. A picture can be bright "
            "overall and still score low here, as long as everything is roughly the same "
            "lightness."
        ),
        technical=(
            "Standard deviation of the preprocessed luminance. After the p1-p99 rescaling it "
            "is bounded by about 0.5, so values are comparable between images. As a second "
            "moment it is sensitive to a few extreme pixels; `robust_contrast` asks "
            "the same question using only the middle 80%. A flat image returns 0.0."
        ),
        references=(
            "[NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)",
            "[Standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(image.std())
