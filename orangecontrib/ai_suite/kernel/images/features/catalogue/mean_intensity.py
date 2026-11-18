import numpy as np

from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class MeanIntensity(Feature):
    name = "mean_intensity"
    description = Description(
        title="Average brightness",
        summary="Mean brightness of the preprocessed image.",
        intuition=(
            "How light or dark the picture is overall, after it has been stretched to its own "
            "darkest and brightest points. A pale overcast sky or a snowfield scores high; a "
            "night street with a few lamps scores low. It cannot tell you whether the "
            "original was shot in sun or shade, only how the tones are balanced inside the "
            "frame."
        ),
        technical=(
            "The arithmetic mean of the preprocessed luminance image. Because preprocessing "
            "rescales every image to its own p1-p99 range, this is relative brightness: it "
            "moves with how tones are distributed, not with exposure. It is a property of the "
            "whole frame and carries no positional information. Same units as "
            "`intensity_std`, different question."
        ),
        references=(
            "[NumPy mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(image.mean())
