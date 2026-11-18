import numpy as np

from .._imaging import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class VerticalSpread(Feature):
    name = "vertical_spread"
    description = Description(
        title="Vertical spread",
        summary="Intensity-weighted standard deviation of the bright mass along y, in image heights.",
        intuition=(
            "Whether the bright matter is tall or squat. A vertical strip of light, such as a "
            "doorway or a tall cloud column, scores high; a low bank of fog scores lower. "
            "Evenly spread brightness again lands near 0.29, the value you get when there is "
            "no single centre to be tight around."
        ),
        technical=(
            "Square root of the intensity-weighted variance of normalised row position, in "
            "units of image height. The y counterpart of `horizontal_spread`, with "
            "the same uniform-image value of 1/sqrt(12) and the same [0, ~0.5] range."
        ),
        references=("[Standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(np.sqrt(max(mass_moments(image).var_y, 0.0)))
