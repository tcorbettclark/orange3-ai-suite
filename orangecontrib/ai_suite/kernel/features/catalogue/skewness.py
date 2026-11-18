import numpy as np

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class Skewness(Feature):
    name = "skewness"
    description = Description(
        title="Brightness skew",
        summary="Whether brightness values lean towards dark or towards light.",
        intuition=(
            "Which side of average most pixels fall on. A small bright cloud in a big dark "
            "sky pushes a few pixels far above the average, so this scores positive. A dark "
            "logo on a white wall pushes a few far below, so it scores negative. A picture "
            "with equal amounts of light and dark, or a smooth ramp, scores near zero."
        ),
        technical=(
            "Moment skewness of the preprocessed luminance, E[(x-mu)^3] / sigma^3, computed "
            "directly from the image. Zero for a symmetric tone distribution, positive when a "
            "small bright region sits in a large dark field, negative in the mirror case. A "
            "flat image returns 0.0. As a third moment it is dominated by a few extreme "
            "pixels; `kurtosis` is the fourth-moment companion."
        ),
        references=(
            "[Skewness](https://en.wikipedia.org/wiki/Skewness)",
            "[NIST: measures of skewness and kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        mean, std = float(image.mean()), float(image.std())
        z = image - mean
        return float((z**3).mean() / std**3)
