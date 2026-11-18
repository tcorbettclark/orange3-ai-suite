import numpy as np

from .._measurements import gradient_magnitude
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class MeanGradientMagnitude(Feature):
    name = "mean_gradient_magnitude"
    description = Description(
        title="Edge strength",
        summary="Mean magnitude of the Sobel gradient: how strongly brightness changes between neighbouring pixels.",
        intuition=(
            "How busy the picture is at a small scale. Foliage, gravel, hair and text score "
            "high; a clear sky, smooth water or a plain wall score low. A single sharp edge "
            "contributes little because the value is averaged over every pixel, so this "
            "measures overall busyness rather than whether any edge exists."
        ),
        technical=(
            "Sobel gradients in x and y over the preprocessed image, combined with "
            "cv2.magnitude and averaged. The 3x3 Sobel operator responds to changes at the "
            "scale of a few pixels, so it is a fine-scale measure; `edge_density` "
            "turns the same gradient field into a fraction, and "
            "`autocorrelation_half_width` gives a typical feature size in pixels."
        ),
        references=(
            "[OpenCV: Sobel derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(gradient_magnitude(image).mean())
