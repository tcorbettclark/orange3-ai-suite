import numpy as np

from .._imaging import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class CentreOfMassY(Feature):
    name = "centre_of_mass_y"
    description = Description(
        title="Vertical centre of mass",
        summary="Intensity-weighted mean vertical position of the bright matter, from 0 at the top to 1 at the bottom.",
        intuition=(
            "The same balancing point as the horizontal version, but measured down the "
            "picture instead of across it. A bright horizon near the top scores low, a bright "
            "foreground near the bottom scores high, and a centred object scores near 0.5. It "
            "is a good companion to the horizontal value: together they locate the bright "
            "mass."
        ),
        technical=(
            "Sum of row mass times normalised row position, divided by total mass, with "
            "position mapped to [0, 1]. A black image returns 0.5. The y counterpart of "
            "`centre_of_mass_x` and, like it, defined here rather than in the profile "
            "features."
        ),
        references=("[Centroid](https://en.wikipedia.org/wiki/Centroid)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return mass_moments(image).com_y
