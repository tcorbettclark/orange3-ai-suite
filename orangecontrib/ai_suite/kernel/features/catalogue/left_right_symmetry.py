"""Bilateral symmetry: left half against the mirrored right half.

Correlation is undefined for a flat pair, so the policy is stated: both halves
constant and equal means perfectly symmetric (1.0); one constant, or both
constant but unequal, means 0.0. Without that guard a near-uniform sky image
would produce a NaN and poison the whole feature vector.
"""

import numpy as np

from .._imaging import mirror_corr
from ..feature import Feature
from ..options import FeatureOptions


class LeftRightSymmetry(Feature):
    name = "left_right_symmetry"
    definition = "Bilateral symmetry: correlation of the left half with the mirrored right half."

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        return mirror_corr(img, axis=1)
