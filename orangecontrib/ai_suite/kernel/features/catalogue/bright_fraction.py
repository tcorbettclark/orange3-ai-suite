"""How much bright stuff, with the threshold taken from the image itself.

The threshold sits half a robust contrast above the image's own median, so a
dim image and a bright image are judged by the same relative standard. A fixed
absolute threshold would silently encode the dataset's lighting.
"""

import numpy as np

from ..feature import Feature, FeatureOptions


class BrightFraction(Feature):
    name = "bright_fraction"
    definition = (
        "Fraction of pixels brighter than the image's own median + "
        "robust_contrast/2 — adaptive 'how much bright stuff'."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        p10, p90 = np.percentile(img, 10), np.percentile(img, 90)
        threshold = float(np.median(img)) + float(p90 - p10) / 2.0
        return float((img > threshold).mean())
