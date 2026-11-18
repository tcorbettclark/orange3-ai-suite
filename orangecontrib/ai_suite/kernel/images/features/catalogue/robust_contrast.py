import numpy as np

from .._measurements import percentile_span
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class RobustContrast(Feature):
    name = "robust_contrast"
    description = Description(
        title="Percentile contrast",
        summary="The spread of the middle 80% of pixel brightness: 90th minus 10th percentile.",
        intuition=(
            "The range of tones most pixels live in, ignoring the darkest and brightest "
            "tenth. A beach photo with one tiny sun glint still scores by the sand and sea "
            "rather than the glint. A hazy scene, with little difference between its shadows "
            "and highlights, scores low even if it contains a few pure black or pure white "
            "specks."
        ),
        technical=(
            "Difference between the 90th and 10th percentiles of the preprocessed luminance: "
            "the width of the central 80% of the tone range. It tolerates up to 10% of pixels "
            "at either extreme, which is why it sets the scale in `bright_fraction`, "
            "and why it is paired with the outlier-sensitive `intensity_std`."
        ),
        references=(
            "[NumPy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)",
            "[Robust statistics](https://en.wikipedia.org/wiki/Robust_statistics)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return percentile_span(image)
