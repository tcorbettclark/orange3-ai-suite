import numpy as np

from .._measurements import local_std_mean
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class FineTexture(Feature):
    name = "fine_texture"
    description = Description(
        title="Fine texture",
        summary="Mean local standard deviation over 5x5 windows: small-scale roughness.",
        intuition=(
            "How grainy the picture is when you look closely. Sand, gravel, grass and fabric "
            "score high; calm water, glass and a clear sky score low. It only sees detail at "
            "the scale of a few pixels, so a large soft cloud can score low even though it is "
            "not uniform over the whole frame."
        ),
        technical=(
            "Mean over the image of the local standard deviation in 5x5 windows, computed as "
            "E[x^2] - E[x]^2 with two box filters. Because the window is a fixed number of "
            "pixels, the feature means what it says only at a fixed working size: with "
            "working_size=None a 64-pixel and a 1024-pixel image of the same scene score "
            "differently. Companion to `coarse_texture`."
        ),
        references=("[Box blur](https://en.wikipedia.org/wiki/Box_blur)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return local_std_mean(image)
