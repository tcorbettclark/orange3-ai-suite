import numpy as np

from .._imaging import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class BluishFraction(Feature):
    name = "bluish_fraction"
    description = Description(
        title="Bluish share",
        summary="Fraction of pixels where blue exceeds red by more than 0.04.",
        intuition=(
            "The deliberate shortcut: how much of the photo is blue. A blue sky or the sea "
            "scores high; a brick wall, autumn leaves or a forest floor scores low. The rule "
            "is so simple that it can answer 'is there sky here?' almost by itself, which is "
            "exactly why these colour features are off by default: they let a model look "
            "clever without learning any shape."
        ),
        technical=(
            "Fraction of pixels where B > R + 0.04 in the absolute colour image, which is "
            "only brought into [0, 1] and resized, never contrast-normalised. Deliberately "
            "naive, and the 0.04 margin is part of the definition. Grayscale input has no "
            "colour and returns 0.0. Enabling the colour features demonstrates shortcut "
            "learning; `greenery_fraction` is its counterpart."
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        red, _, blue = channels
        return float((blue > red + 0.04).mean())
