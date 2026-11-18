import numpy as np

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class MeanChroma(Feature):
    name = "mean_chroma"
    description = Description(
        title="Colourfulness",
        summary="Mean span between the largest and smallest channel value per pixel.",
        intuition=(
            "How colourful the picture is. Vivid flowers, painted signs and a rainbow score "
            "high; concrete, fog, snow and black-and-white photographs score low. It is about "
            "how far the three colours differ in each pixel, not about which colour is "
            "present, so a strong blue and a strong red both count."
        ),
        technical=(
            "Mean over pixels of max(R, G, B) - min(R, G, B): the chroma of each pixel, or "
            "HSV saturation without the division by the maximum channel. Scale-dependent on "
            "the absolute colour image, which is only brought into [0, 1] before measuring. "
            "Grayscale returns 0.0. Named for what is computed rather than for HSV "
            "saturation, which would divide by the maximum channel."
        ),
        references=("[Chrominance](https://en.wikipedia.org/wiki/Chrominance)",),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        planes = np.stack(channels)
        return float((planes.max(axis=0) - planes.min(axis=0)).mean())
