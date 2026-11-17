"""Vertical mirror symmetry: top half against the mirrored bottom half."""

import numpy as np

from .._imaging import mirror_corr
from ..feature import Feature
from ..options import FeatureOptions


class TopBottomSymmetry(Feature):
    name = "top_bottom_symmetry"
    definition = (
        "Vertical mirror symmetry: correlation of the top half with the mirrored bottom half."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return mirror_corr(img, axis=0)
