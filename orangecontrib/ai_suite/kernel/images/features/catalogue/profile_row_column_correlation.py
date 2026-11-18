import numpy as np

from .._conventions import safe_corr
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class ProfileRowColumnCorrelation(Feature):
    name = "profile_row_column_correlation"
    description = Description(
        title="Row/column agreement",
        summary="Correlation of the row-mean and column-mean brightness profiles.",
        intuition=(
            "Do the horizontal and vertical summaries of the picture agree? A single round "
            "blob scores high, because both summaries peak in the middle. A vertical stripe "
            "scores low: the columns are busy, the rows are barely changing. A completely "
            "flat picture scores zero, because there is nothing for the two summaries to "
            "agree or disagree about."
        ),
        technical=(
            "Pearson correlation of the row-mean and column-mean brightness profiles. The "
            "profiles are short (one value per row and per column), so the correlation is of "
            "shape, not of pixels. A flat profile has no structure to correlate and returns "
            "0.0 rather than a NaN, using the shared flat-input rule in "
            "_conventions.safe_corr."
        ),
        references=(
            "[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return safe_corr(image.mean(axis=1), image.mean(axis=0), flat_policy="zero")
