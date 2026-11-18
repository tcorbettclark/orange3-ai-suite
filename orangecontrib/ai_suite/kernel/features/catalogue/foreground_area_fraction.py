"""How much of the frame is above the Otsu threshold: ink, or cloud cover."""

import numpy as np

from .._imaging import otsu_foreground
from ..feature import Feature
from ..options import FeatureOptions


class ForegroundAreaFraction(Feature):
    name = "foreground_area_fraction"
    definition = "Ink / cloud coverage: fraction of pixels above the Otsu threshold."

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(otsu_foreground(image).mean())
