import numpy as np

from .._imaging import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class CentreOfMassX(Feature):
    name = "centre_of_mass_x"
    description = Description(
        title="Horizontal centre of mass",
        summary="Intensity-weighted mean horizontal position of the bright matter, from 0 on the left to 1 on the right.",
        intuition=(
            "If you cut the picture into vertical slices, weighted each by how bright it is, "
            "and balanced them on a finger, this is where the finger would go. A sun low on "
            "the left scores near 0.2, a centred subject near 0.5, a bright window on the "
            "right near 0.8. Dark areas count for almost nothing."
        ),
        technical=(
            "Sum of column mass times normalised column position, divided by total mass. "
            "Position is the pixel index mapped to [0, 1], so the value is independent of "
            "working size. A black image has no mass and returns 0.5. The single home of "
            "horizontal centre of mass: the profile features do not restate it."
        ),
        references=("[Centroid](https://en.wikipedia.org/wiki/Centroid)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return mass_moments(image).com_x
