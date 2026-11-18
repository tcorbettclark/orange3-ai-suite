import numpy as np

from .._conventions import EPS
from .._measurements import block_mean_variance, local_std_mean
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class CoarseToFineTextureRatio(Feature):
    name = "coarse_to_fine_texture_ratio"
    description = Description(
        title="Texture scale balance",
        summary="Coarse texture divided by coarse plus fine: 0.5 is scale-neutral, high means big lumps dominate.",
        intuition=(
            "Which size of detail rules the picture. Puffy cumulus cloud, made of big soft "
            "blobs, scores high; wispy cirrus, made of thin threads, scores low; a scene with "
            "both scales mixed scores near the middle. It is a balance rather than an amount, "
            "so a busy picture and a calm one can score the same if their detail sits at the "
            "same scale."
        ),
        technical=(
            "coarse / (coarse + fine + EPS), bounded to [0, 1]. The bounded form is used "
            "rather than coarse / fine, which divides two small numbers and is "
            "ill-conditioned exactly when the image is smooth. Both ingredients come from "
            "`coarse_texture` and `fine_texture`, so the ratio inherits their "
            "fixed pixel windows."
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        coarse = block_mean_variance(image)
        fine = local_std_mean(image)
        return coarse / (coarse + fine + EPS)
