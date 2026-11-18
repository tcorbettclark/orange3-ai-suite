import numpy as np

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class Kurtosis(Feature):
    name = "kurtosis"
    description = Description(
        title="Histogram peakedness",
        summary="How sharply the brightness histogram peaks: high when a few tones dominate, low when every shade appears.",
        intuition=(
            "Imagine sorting every pixel from black to white and drawing a bar chart of how "
            "many sit at each brightness. A white wall with a small dark logo gives one huge "
            "spike of white and a thin tail of black - a tall, narrow shape, which scores "
            "high. A busy street at dusk, with every shade from deep shadow to lamplight, "
            "spreads out evenly and scores near zero. It reacts to tone, not layout: "
            "shuffling the pixels changes nothing."
        ),
        technical=(
            "Excess kurtosis of the preprocessed luminance, E[(x-mu)^4] / sigma^4 - 3, so 0 "
            "is the Gaussian reference and -2 the uniform distribution. A shape statistic: "
            "invariant to translation and to pixel order, so it says nothing about where "
            "anything is. The fourth moment makes it more sensitive to a few extreme pixels "
            "than `skewness`; `robust_contrast` is the resistant alternative. "
            "A flat image returns 0.0."
        ),
        references=(
            "[Kurtosis](https://en.wikipedia.org/wiki/Kurtosis)",
            "[NIST: measures of skewness and kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        mean, std = float(image.mean()), float(image.std())
        z = image - mean
        return float((z**4).mean() / std**4 - 3.0)
