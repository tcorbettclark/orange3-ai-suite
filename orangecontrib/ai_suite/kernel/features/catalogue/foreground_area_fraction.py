import numpy as np

from .._imaging import otsu_foreground
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class ForegroundAreaFraction(Feature):
    name = "foreground_area_fraction"
    description = Description(
        title="Bright-area fraction (Otsu)",
        summary="Fraction of pixels above the Otsu threshold: the bright foreground.",
        intuition=(
            "How much of the frame would be left if you kept only the bright part. A thick "
            "sheet of cloud covers a lot and scores high; a thin wisp covers little and "
            "scores low. A photo of a dark object on a bright table scores high instead, "
            "because Otsu splits the picture at its brightest gap and calls the larger bright "
            "side foreground."
        ),
        technical=(
            "Otsu's threshold is computed on the image min-max scaled to 8 bits, then mapped "
            "back to the original range (Otsu is invariant under affine rescaling, so this is "
            "exact). The fraction of pixels above it is reported. A flat image has no "
            "threshold and returns 0.0. Related: `bright_fraction` uses a "
            "median-plus-span cut instead of Otsu."
        ),
        references=("[Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method)",),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(otsu_foreground(image).mean())
