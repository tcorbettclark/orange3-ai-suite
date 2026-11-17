"""Where the bright mass sits horizontally.

This is the one home of horizontal centre of mass: the row/column profile
features deliberately do not restate it from their projections.
"""

import numpy as np

from .._primitives import mass_moments
from ..feature import Feature, FeatureOptions


class CentreOfMassX(Feature):
    name = "centre_of_mass_x"
    definition = "Left/right placement of the bright mass in [0, 1]; 0.5 = centred."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return mass_moments(img).com_x
