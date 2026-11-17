"""The deliberate shortcut: how much of the photo is blue.

"Fraction of bluish pixels" answers "is there sky in this photo?" almost by
itself, so a model given this feature can score well without learning any
spatial structure. That is exactly why it is opt-in: switching the colour
features on is the shortcut-learning demonstration, not an improvement.
"""

import numpy as np

from .._primitives import colour_channels
from ..feature import Feature, FeatureOptions


class BluishFraction(Feature):
    name = "bluish_fraction"
    definition = (
        "The shortcut: fraction of pixels bluer than red + 0.04 - sky "
        "presence, which answers 'is there sky in this photo' trivially. "
        "Enabling the colour features demonstrates shortcut learning."
    )
    needs_colour = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        channels = colour_channels(img)
        if channels is None:
            return 0.0
        red, _, blue = channels
        return float((blue > red + 0.04).mean())
