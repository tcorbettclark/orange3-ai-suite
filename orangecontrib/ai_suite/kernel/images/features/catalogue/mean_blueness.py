import numpy as np

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class MeanBlueness(Feature):
    name = "mean_blueness"
    description = Description(
        title="Blue-red balance",
        summary="Mean of blue minus red across all pixels.",
        intuition=(
            "How far the whole picture leans to blue rather than red. A clear sky scores "
            "strongly positive, an autumn wood or a brick wall negative, and a grey street "
            "close to zero. It is an average over the frame, so one blue corner barely moves "
            "it while a blue cast over everything does."
        ),
        technical=(
            "Mean of (B - R) over the absolute colour image, in [-1, 1]. Unlike "
            "`bluish_fraction` it is not a test per pixel, so a small very blue area "
            "is diluted by the rest of the frame. Grayscale input returns 0.0. One of the "
            "four deliberately naive colour features."
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        red, _, blue = channels
        return float((blue - red).mean())
