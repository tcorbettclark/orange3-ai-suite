"""Do the horizontal and vertical structures agree?

Lumpy images correlate high, streaky images low. A flat profile carries no
structure to agree or disagree about, so it returns 0.0 rather than a NaN.
"""

import numpy as np

from .._conventions import safe_corr
from ..feature import Feature
from ..options import FeatureOptions


class ProfileRowColumnCorrelation(Feature):
    name = "profile_row_column_correlation"
    definition = (
        "Do horizontal and vertical structures agree? Correlation of the row- "
        "and column-mean profiles; 0.0 when either is flat (lumpy vs. streaky)."
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return safe_corr(image.mean(axis=1), image.mean(axis=0), flat_policy="zero")
