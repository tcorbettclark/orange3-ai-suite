"""How much of the image is edgy, with the threshold taken from the image.

The threshold is a fraction of the image's *own* 99th-percentile gradient, not
a fixed value: a fixed percentile threshold would return the same density for
every image, which is the definition of a useless feature. The fraction is
configurable (``edge_density_frac``), so the definition says "configured"
rather than naming a number.
"""

import numpy as np

from .._conventions import EPS
from .._imaging import gradient_magnitude
from ..feature import Feature
from ..options import FeatureOptions


class EdgeDensity(Feature):
    name = "edge_density"
    definition = (
        "Fraction of 'edgy' pixels: gradient at least a configured fraction "
        "(default half) of the image's own p99 gradient (adaptive)."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        magnitude = gradient_magnitude(img)
        p99 = float(np.percentile(magnitude, 99))
        if p99 < EPS:
            return 0.0
        return float((magnitude >= opts.edge_density_frac * p99).mean())
