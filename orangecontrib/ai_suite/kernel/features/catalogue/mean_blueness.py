"""Overall blue-red balance: the sky colour cast, averaged.

The definition is deliberately naive; that naivety is the point.
"""

import numpy as np

from .._primitives import colour_channels
from ..feature import Feature, FeatureOptions


class MeanBlueness(Feature):
    name = "mean_blueness"
    definition = "Overall sky colour balance: mean(B - R) in [-1, 1]."
    needs_colour = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        channels = colour_channels(img)
        if channels is None:
            return 0.0
        red, _, blue = channels
        return float((blue - red).mean())
