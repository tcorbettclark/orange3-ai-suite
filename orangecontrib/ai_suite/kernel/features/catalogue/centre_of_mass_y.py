"""Where the bright mass sits vertically."""

import numpy as np

from .._imaging import mass_moments
from ..feature import Feature
from ..options import FeatureOptions


class CentreOfMassY(Feature):
    name = "centre_of_mass_y"
    definition = "Top/bottom placement of the bright mass in [0, 1]; 0.5 = centred."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return mass_moments(img).com_y
