"""Where the bright mass sits horizontally.

This is the one home of horizontal centre of mass: the row/column profile
features deliberately do not restate it from their projections.
"""

import numpy as np

from .._imaging import mass_moments
from ..feature import Feature
from ..options import FeatureOptions


class CentreOfMassX(Feature):
    name = "centre_of_mass_x"
    definition = "Left/right placement of the bright mass in [0, 1]; 0.5 = centred."

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return mass_moments(image).com_x
