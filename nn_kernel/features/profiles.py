"""Group B — profile summaries (12 features): dense generalised line probes.

The full row projection (mean intensity per row) and column projection are
built and summarised, so every line contributes — sparse probes are
position-sensitive (spec §2 weakness table).

Definitions
-----------
- trend: linear-fit slope over the whole profile, scaled by its length =
  net intensity change from one side to the other, in intensity units
  (bounded roughly [-1, 1]). Sign carries "brighter at the bottom/top".
- roughness: total variation / mean level (dimensionless lumpiness).
- peaks: number of interior maxima with prominence >=
  ``peak_prominence_frac`` x dynamic range (agreed Q1). Profiles are padded
  with their minimum on both ends so a maximum sitting at the image border
  still counts as a peak; such padding can never create a false peak.
- centre of mass: intensity-weighted mean position, normalised to [0, 1]
  across the profile.
- diagonal / anti-diagonal profiles: dense projections along diagonal bands
  via ``ndarray.diagonal`` (agreed Q2, option b) — exact, no interpolation.
- profile correlation: Pearson correlation of the row and column profiles;
  0.0 if either is flat (no structure to agree or disagree).
- median up-crossing count: for every line (row, then column), count
  up-crossings of that line's own median level; take the median over lines;
  average the row and column medians. Parameter-free cousin of peak counting.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .options import FeatureOptions

_EPS = 1e-12

FEATURE_NAMES = (
    "row_profile_trend",
    "row_profile_roughness",
    "row_profile_peaks",
    "row_profile_centre_of_mass",
    "column_profile_trend",
    "column_profile_roughness",
    "column_profile_peaks",
    "column_profile_centre_of_mass",
    "diagonal_profile_peaks",
    "anti_diagonal_profile_peaks",
    "profile_row_column_correlation",
    "median_upcrossing_count",
)

DEFINITIONS = {
    "row_profile_trend": (
        "Net intensity change from image top to bottom (linear fit of the "
        "row-mean profile scaled by its length); positive = brighter at the bottom."
    ),
    "row_profile_roughness": (
        "How lumpy the image is vertically: total variation of the row-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    ),
    "row_profile_peaks": (
        "Number of horizontal bands of bright material: maxima of the row-mean "
        "profile with prominence at least the configured fraction of its "
        "dynamic range; border maxima included."
    ),
    "row_profile_centre_of_mass": (
        "Vertical position of the bright mass in [0, 1]; 0.5 = centred."
    ),
    "column_profile_trend": (
        "Net intensity change from image left to right (linear fit of the "
        "column-mean profile scaled by its length); positive = brighter on the right."
    ),
    "column_profile_roughness": (
        "How lumpy the image is horizontally: total variation of the column-mean "
        "profile divided by its mean level; 0 = perfectly smooth."
    ),
    "column_profile_peaks": (
        "Number of vertical bands of bright material: maxima of the column-mean "
        "profile with prominence at least the configured fraction of its "
        "dynamic range; border maxima included."
    ),
    "column_profile_centre_of_mass": (
        "Horizontal position of the bright mass in [0, 1]; 0.5 = centred."
    ),
    "diagonal_profile_peaks": (
        "Number of bright diagonal bands along the main diagonal, from a dense "
        "projection over all diagonals (no interpolation)."
    ),
    "anti_diagonal_profile_peaks": (
        "Number of bright diagonal bands along the anti-diagonal, from a dense "
        "projection over all diagonals (no interpolation)."
    ),
    "profile_row_column_correlation": (
        "Do horizontal and vertical structures agree? Correlation of the row- "
        "and column-mean profiles; 0.0 when either is flat (lumpy vs. streaky)."
    ),
    "median_upcrossing_count": (
        "How often pixel lines cross their own median level — parameter-free "
        "cousin of peak counting (median over rows and columns, averaged)."
    ),
}


def _profile(img: np.ndarray, axis: int) -> np.ndarray:
    return img.mean(axis=axis)


def _trend(prof: np.ndarray) -> float:
    n = len(prof)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    slope = np.polyfit(x, prof.astype(np.float64), 1)[0]
    return float(slope * n)


def _roughness(prof: np.ndarray) -> float:
    tv = float(np.abs(np.diff(prof)).sum())
    return tv / (float(prof.mean()) + _EPS)


def _peak_count(prof: np.ndarray, opts: FeatureOptions) -> float:
    rng = float(prof.max() - prof.min())
    if rng < _EPS:
        return 0.0
    padded = np.concatenate([[prof.min()], prof, [prof.min()]])
    peaks, _ = find_peaks(padded, prominence=rng * opts.peak_prominence_frac)
    return float(len(peaks))


def _centre_of_mass(prof: np.ndarray) -> float:
    w = np.clip(prof.astype(np.float64), 0.0, None)
    n = len(prof)
    if w.sum() < _EPS:
        return (n - 1) / 2.0 / max(n - 1, 1)
    return float(np.average(np.arange(n), weights=w) / (n - 1))


def _diagonal_profile(img: np.ndarray, anti: bool) -> np.ndarray:
    m = img[:, ::-1] if anti else img
    h, w = img.shape
    return np.asarray(
        [m.diagonal(offset=d).mean() for d in range(-(h - 1), w)],
        dtype=np.float64,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two profiles, flat-resistant.

    The guard is on centred amplitude, not std: float32 summation noise can
    leave a structurally constant profile with std ~1e-8, which would poison
    corrcoef with 0/0. Flat profiles carry no structure to agree or disagree
    about -> 0.0 (documented semantics of the correlation feature).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    am = a - a.mean()
    bm = b - b.mean()
    amp = max(float(np.abs(am).max()), float(np.abs(bm).max()))
    if amp < 1e-6:
        return 0.0
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if den == 0.0 or not np.isfinite(den):
        return 0.0
    # elementwise-sum, NOT `am @ bm`: matmul dispatch hits spurious BLAS
    # warnings on some numpy builds (seen: 2.1.3/macOS arm) for identical math
    return float(np.clip((am * bm).sum() / den, -1.0, 1.0))


def _level_upcrossings(line: np.ndarray) -> float:
    level = np.median(line)
    above = line > level
    return float((above[1:] & ~above[:-1]).sum())


def _median_upcrossing(img: np.ndarray) -> float:
    rows = [_level_upcrossings(img[i, :]) for i in range(img.shape[0])]
    cols = [_level_upcrossings(img[:, j]) for j in range(img.shape[1])]
    return float((float(np.median(rows)) + float(np.median(cols))) / 2.0)


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    row = _profile(img, axis=1)
    col = _profile(img, axis=0)
    return {
        "row_profile_trend": _trend(row),
        "row_profile_roughness": _roughness(row),
        "row_profile_peaks": _peak_count(row, opts),
        "row_profile_centre_of_mass": _centre_of_mass(row),
        "column_profile_trend": _trend(col),
        "column_profile_roughness": _roughness(col),
        "column_profile_peaks": _peak_count(col, opts),
        "column_profile_centre_of_mass": _centre_of_mass(col),
        "diagonal_profile_peaks": _peak_count(_diagonal_profile(img, anti=False), opts),
        "anti_diagonal_profile_peaks": _peak_count(
            _diagonal_profile(img, anti=True), opts
        ),
        "profile_row_column_correlation": _safe_corr(row, col),
        "median_upcrossing_count": _median_upcrossing(img),
    }