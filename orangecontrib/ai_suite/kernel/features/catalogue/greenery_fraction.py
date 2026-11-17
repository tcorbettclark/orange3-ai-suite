"""Ground clutter: how much of the frame is greener than red or blue.

Terrain, trees and grass in the lower half of a sky photo. The complement of
``bluish_fraction`` in practice, which is why the pair separates "sky" from
"not sky" so cheaply - and why both are excluded from the default selection.
"""

import numpy as np

from .._primitives import colour_channels
from ..feature import Feature, FeatureOptions


class GreeneryFraction(Feature):
    name = "greenery_fraction"
    definition = (
        "Ground-clutter detector: fraction of pixels greener than both red "
        "and blue (grass, trees, terrain)."
    )
    needs_colour = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        channels = colour_channels(img)
        if channels is None:
            return 0.0
        red, green, blue = channels
        return float(((green > red) & (green > blue)).mean())
