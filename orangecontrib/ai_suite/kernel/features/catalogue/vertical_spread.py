"""Tall or squat: the mass's standard deviation along y, in image heights."""

import numpy as np

from .._primitives import mass_moments
from ..feature import Feature, FeatureOptions


class VerticalSpread(Feature):
    name = "vertical_spread"
    definition = (
        "Tall vs. squat bright mass: intensity-weighted std along y, divided by image height."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return float(np.sqrt(max(mass_moments(img).var_y, 0.0)))
