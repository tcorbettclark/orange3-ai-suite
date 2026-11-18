"""Where the texture lives: coarse as a fraction of total.

The bounded form ``coarse / (coarse + fine)`` is used instead of
``coarse / fine``, which divides two small numbers and is ill-conditioned
exactly when images are smooth. 0.5 is scale-neutral; puffy cumulus scores high
and wispy cirrus low, which is why this carries so much cloud signal.
"""

import numpy as np

from .._conventions import EPS
from .._imaging import block_mean_variance, local_std_mean
from ..feature import Feature
from ..options import FeatureOptions


class CoarseToFineTextureRatio(Feature):
    name = "coarse_to_fine_texture_ratio"
    definition = (
        "Where the texture lives: coarse texture as a fraction of total "
        "texture (coarse/(coarse+fine), [0,1], 0.5 = scale-neutral); puffy "
        "cumulus scores high, wispy cirrus low."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        coarse = block_mean_variance(image)
        fine = local_std_mean(image)
        return coarse / (coarse + fine + EPS)
