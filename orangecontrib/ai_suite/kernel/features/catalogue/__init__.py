"""The feature catalogue: one module, one class, one feature.

Importing this package imports every feature, which is deliberate: each class
validates itself when it is created (see :mod:`..feature`), so a malformed
feature fails here rather than during extraction.

``CATALOGUE`` is the roster and the column order - the only place either is
defined. It is written out rather than discovered by scanning the directory: a
module that nothing imports would otherwise be silently invisible, which is a
worse failure than a stale list. The tests assert that the roster and the
directory agree.

Order here is column order and nothing else. Which features belong together is
a grouping concern (:mod:`..groups`), so neither this file nor the classes it
lists may group the features into families: a second statement of the grouping
would only be something else to go stale.
"""

from ..feature import Feature
from .anti_diagonal_profile_peaks import AntiDiagonalProfilePeaks
from .autocorrelation_half_width import AutocorrelationHalfWidth
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
from .mean_blueness import MeanBlueness
from .mean_gradient_magnitude import MeanGradientMagnitude
from .mean_intensity import MeanIntensity
from .mean_saturation import MeanSaturation
from .profile_median_crossing_count import ProfileMedianCrossingCount
from .profile_row_column_correlation import ProfileRowColumnCorrelation
from .quadrant_self_similarity import QuadrantSelfSimilarity
from .radial_gradient import RadialGradient
from .robust_contrast import RobustContrast
from .row_profile_peaks import RowProfilePeaks
from .row_profile_roughness import RowProfileRoughness
from .row_profile_trend import RowProfileTrend
from .skewness import Skewness
from .stroke_crossings_median_level import StrokeCrossingsMedianLevel
from .top_bottom_symmetry import TopBottomSymmetry
from .vertical_spread import VerticalSpread

CATALOGUE: tuple[Feature, ...] = (
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
    StrokeCrossingsMedianLevel(),
    LeftRightSymmetry(),
    TopBottomSymmetry(),
    QuadrantSelfSimilarity(),
    BluishFraction(),
    MeanBlueness(),
    MeanSaturation(),
    GreeneryFraction(),
)

__all__ = ["CATALOGUE"]
