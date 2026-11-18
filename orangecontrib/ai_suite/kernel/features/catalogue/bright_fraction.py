import numpy as np

from .._imaging import percentile_span
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class BrightFraction(Feature):
    name = "bright_fraction"
    description = Description(
        title="Bright-area share",
        summary="Fraction of pixels brighter than the image's own median plus half its percentile contrast.",
        intuition=(
            "How much of the picture stands out as noticeably brighter than the typical "
            "pixel. Two small clouds in a blue sky score low; a sky half covered in white "
            "cloud scores high. One small bright lamp also scores low, because a few pixels "
            "are not a large share. The cut-off is taken from each picture, so dim and bright "
            "images are judged alike."
        ),
        technical=(
            "Threshold at median + (p90 - p10) / 2, then the fraction of pixels above it. The "
            "offset is half the central tone range, so the cut-off follows the image's own "
            "contrast instead of a fixed level. A flat image has a zero span and returns 0.0. "
            "Compare `foreground_area_fraction`, which uses an Otsu threshold "
            "instead."
        ),
        references=(
            "[NumPy median](https://numpy.org/doc/stable/reference/generated/numpy.median.html)",
            "[Median](https://en.wikipedia.org/wiki/Median)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        threshold = float(np.median(image)) + percentile_span(image) / 2.0
        return float((image > threshold).mean())
