"""Tall or squat: the mass's standard deviation along y, in image heights."""

import numpy as np

from .._imaging import mass_moments
from ..feature import Feature
from ..options import FeatureOptions


class VerticalSpread(Feature):
    name = "vertical_spread"
    definition = (
        "Tall vs. squat bright mass: intensity-weighted std along y, divided by image height."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(np.sqrt(max(mass_moments(image).var_y, 0.0)))
