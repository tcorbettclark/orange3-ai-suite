"""Wide or narrow: the mass's standard deviation along x, in image widths.

Divided by the image width, so it is comparable between a 64-pixel and a
1024-pixel working size. Uniform mass gives ~0.29.
"""

import numpy as np

from .._primitives import mass_moments
from ..feature import Feature, FeatureOptions


class HorizontalSpread(Feature):
    name = "horizontal_spread"
    definition = (
        "Wide vs. narrow bright mass: intensity-weighted std along x, divided by image width."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(np.sqrt(max(mass_moments(img).var_x, 0.0)))
