"""The feature catalogue: every feature in column order, by name and by position.

Each feature is a class in its own module and one instance in ``_CATALOGUE``, whose
order is the column order and the only definition of it.

Importing this package imports every feature, and each class validates itself when
it is created, so a malformed feature fails on import rather than during extraction.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..feature import Feature
from .anti_diagonal_profile_peaks import AntiDiagonalProfilePeaks
from .autocorrelation_half_width import AutocorrelationHalfWidth
from .blue_minus_red_mean import BlueMinusRedMean
from .bluish_fraction import BluishFraction
from .bright_fraction import BrightFraction
from .centre_of_mass_x import CentreOfMassX
from .centre_of_mass_y import CentreOfMassY
from .coarse_texture import CoarseTexture
from .coarse_to_fine_texture_ratio import CoarseToFineTextureRatio
from .column_profile_peaks import ColumnProfilePeaks
from .column_profile_roughness import ColumnProfileRoughness
from .column_profile_trend import ColumnProfileTrend
from .diagonal_profile_peaks import DiagonalProfilePeaks
from .edge_density import EdgeDensity
from .elongation import Elongation
from .fine_texture import FineTexture
from .foreground_area_fraction import ForegroundAreaFraction
from .greenery_fraction import GreeneryFraction
from .hole_count import HoleCount
from .horizontal_spread import HorizontalSpread
from .intensity_std import IntensityStd
from .kurtosis import Kurtosis
from .left_right_symmetry import LeftRightSymmetry
from .mean_chroma import MeanChroma
from .mean_gradient_magnitude import MeanGradientMagnitude
from .mean_intensity import MeanIntensity
from .profile_median_crossing_count import ProfileMedianCrossingCount
from .profile_row_column_correlation import ProfileRowColumnCorrelation
from .quadrant_self_similarity import QuadrantSelfSimilarity
from .radial_gradient import RadialGradient
from .robust_contrast import RobustContrast
from .row_profile_peaks import RowProfilePeaks
from .row_profile_roughness import RowProfileRoughness
from .row_profile_trend import RowProfileTrend
from .skewness import Skewness
from .stroke_crossing_count import StrokeCrossingCount
from .top_bottom_symmetry import TopBottomSymmetry
from .vertical_spread import VerticalSpread

_CATALOGUE: tuple[Feature, ...] = (
    MeanIntensity(),
    IntensityStd(),
    RobustContrast(),
    Skewness(),
    Kurtosis(),
    BrightFraction(),
    RowProfileTrend(),
    RowProfileRoughness(),
    RowProfilePeaks(),
    ColumnProfileTrend(),
    ColumnProfileRoughness(),
    ColumnProfilePeaks(),
    DiagonalProfilePeaks(),
    AntiDiagonalProfilePeaks(),
    ProfileRowColumnCorrelation(),
    ProfileMedianCrossingCount(),
    CentreOfMassX(),
    CentreOfMassY(),
    HorizontalSpread(),
    VerticalSpread(),
    Elongation(),
    RadialGradient(),
    MeanGradientMagnitude(),
    EdgeDensity(),
    FineTexture(),
    CoarseTexture(),
    CoarseToFineTextureRatio(),
    AutocorrelationHalfWidth(),
    HoleCount(),
    ForegroundAreaFraction(),
    StrokeCrossingCount(),
    LeftRightSymmetry(),
    TopBottomSymmetry(),
    QuadrantSelfSimilarity(),
    BluishFraction(),
    BlueMinusRedMean(),
    MeanChroma(),
    GreeneryFraction(),
)


def _build(measures: Iterable[Feature]) -> dict[str, Feature]:
    table: dict[str, Feature] = {}
    summaries: dict[str, str] = {}
    titles: dict[str, str] = {}
    problems: list[str] = []
    for feature in measures:
        if feature.name in table:
            problems.append(f"duplicate feature name {feature.name!r}")
        if feature.description.summary in summaries:
            problems.append(
                f"{feature.name!r} repeats the summary of "
                f"{summaries[feature.description.summary]!r}"
            )
        if feature.description.title in titles:
            problems.append(
                f"{feature.name!r} repeats the title of {titles[feature.description.title]!r}"
            )
        summaries[feature.description.summary] = feature.name
        titles[feature.description.title] = feature.name
        table[feature.name] = feature
    if problems:
        raise RuntimeError("; ".join(problems))
    return table


_BY_NAME: dict[str, Feature] = _build(_CATALOGUE)

#: Column position of each feature, derived once from the roster order.
POSITION: dict[str, int] = {name: i for i, name in enumerate(_BY_NAME)}


def all_features() -> tuple[Feature, ...]:
    """All features in stable column order."""
    return tuple(_BY_NAME.values())


def all_feature_names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def all_count_feature_names() -> tuple[str, ...]:
    """Names of the integer-valued count features, in column order."""
    return tuple(feature.name for feature in _BY_NAME.values() if feature.is_count)


def is_known(name: str) -> bool:
    """Whether a feature name exists in the catalogue."""
    return name in _BY_NAME


def get_feature(name: str) -> Feature:
    """One feature by name; ``ValueError`` when the name is not in the catalogue."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown feature {name!r}; see all_feature_names() ({len(_BY_NAME)} available)"
        ) from None


__all__ = [
    "all_count_feature_names",
    "all_feature_names",
    "all_features",
    "get_feature",
]
