import numpy as np

from .._measurements import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class HorizontalSpread(Feature):
    name = "horizontal_spread"
    description = Description(
        title="Horizontal spread",
        summary="Intensity-weighted standard deviation of the bright mass along x, in image widths.",
        intuition=(
            "Whether the bright matter is wide or narrow. A long horizon line spreads across "
            "the frame and scores around 0.3; a small sun is compact and scores closer to "
            "0.05. A picture whose brightness is spread evenly over the whole frame scores "
            "about 0.29, so values near that mean there is no particular horizontal centre."
        ),
        technical=(
            "Square root of the intensity-weighted variance of normalised column position. "
            "Both position and spread are divided by the image side, so the value lives in "
            "[0, ~0.5] and is comparable across working sizes; a uniform image gives "
            "1/sqrt(12) = 0.29. Same weighting as `centre_of_mass_x`."
        ),
        references=("[Standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(np.sqrt(max(mass_moments(image).var_x, 0.0)))
