import numpy as np

from .._conventions import EPS
from .._imaging import gradient_magnitude
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class EdgeDensity(Feature):
    name = "edge_density"
    description = Description(
        title="Edgy-pixel share",
        summary="Fraction of pixels whose gradient is at least a configured fraction of the image's own 99th percentile gradient.",
        intuition=(
            "How much of the picture is edge rather than flat area. A tree line against the "
            "sky is a thin edge and scores low, even though it is sharp; a brick wall, a "
            "crowd or grass is nearly all edge and scores high. The cut-off comes from each "
            "picture's own strongest edges, so a soft, low-contrast photo is judged on its "
            "own terms."
        ),
        technical=(
            "Sobel gradient magnitude, thresholded at options.edge_density_frac (default 0.5) "
            "times the image's 99th percentile gradient, and averaged. The fraction is "
            "configurable, so the definition says 'configured', not a number. A flat image "
            "has no gradient and returns 0.0. A fixed percentile threshold would return the "
            "same density for every image, which is why the threshold is a fraction of each "
            "image's own p99."
        ),
        references=(
            "[OpenCV: Sobel derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        magnitude = gradient_magnitude(image)
        p99 = float(np.percentile(magnitude, 99))
        if p99 < EPS:
            return 0.0
        return float((magnitude >= options.edge_density_frac * p99).mean())
