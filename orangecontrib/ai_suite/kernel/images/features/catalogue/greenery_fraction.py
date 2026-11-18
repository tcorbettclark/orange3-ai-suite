import numpy as np

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class GreeneryFraction(Feature):
    name = "greenery_fraction"
    description = Description(
        title="Green share",
        summary="Fraction of pixels where green exceeds both red and blue.",
        intuition=(
            "How much of the frame is vegetation. Grass, trees, lawns and fields score high; "
            "sky, water, pavement and sand score low. It is the opposite of the bluish share "
            "in practice, so the pair separates a green landscape from a blue sky almost for "
            "free - and that is exactly the shortcut the colour features are there to "
            "demonstrate."
        ),
        technical=(
            "Fraction of pixels with G > R and G > B in the absolute colour image. A strict "
            "two-way test with no margin, unlike the 0.04 offset in "
            "`bluish_fraction`. Grayscale input returns 0.0. The fourth deliberately "
            "naive colour feature, off by default with the rest of the group."
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        red, green, blue = channels
        return float(((green > red) & (green > blue)).mean())
