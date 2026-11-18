"""Shared numeric conventions: what "zero", "flat" and "net change" mean.

These are rules, not measurements. More than one feature needs each of them, so
one copy means the epsilon, the flat-input policy and the across-a-profile
scaling cannot drift apart between features. Image measurements live in
:mod:`_imaging`.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12  # "numerically zero": division, range and empty-mass guards

FLAT_AMPLITUDE = 1e-6  # centred amplitude below this counts as a flat input

_FLAT_POLICIES = ("zero", "equal_is_one")


def is_flat(array: np.ndarray) -> bool:
    """Whether an array carries no structure, by centred amplitude.

    The guard is on amplitude rather than std: float32 summation noise can leave
    a structurally constant profile with std ~1e-8, which would poison a
    correlation with 0/0. One tolerance for every feature that has to decide.
    """
    values = np.asarray(array, dtype=np.float64).ravel()
    return float(np.abs(values - values.mean()).max()) < FLAT_AMPLITUDE


def safe_corr(a: np.ndarray, b: np.ndarray, *, flat_policy: str) -> float:
    """Pearson correlation with an explicit policy for flat (constant) inputs.

    A flat array has zero centred amplitude, so its correlation is undefined.
    The caller states what that means for its feature:

    - ``"zero"``: a flat side carries no structure to agree or disagree about,
      so the result is 0.0 (profile correlation).
    - ``"equal_is_one"``: both sides constant and equal is perfect agreement,
      so 1.0; one side constant, or both constant but unequal, is 0.0
      (mirror symmetry of a uniform region).

    The guard is on centred amplitude, not std: float32 summation noise can
    leave a structurally constant profile with std ~1e-8, which would poison
    the correlation with 0/0.
    """
    if flat_policy not in _FLAT_POLICIES:
        raise ValueError(f"unknown flat policy {flat_policy!r}; valid: {_FLAT_POLICIES}")
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    am = a - a.mean()
    bm = b - b.mean()
    flat_a = float(np.abs(am).max()) < FLAT_AMPLITUDE
    flat_b = float(np.abs(bm).max()) < FLAT_AMPLITUDE
    if flat_a or flat_b:
        if flat_policy == "equal_is_one" and flat_a and flat_b:
            return 1.0 if abs(float(a.mean()) - float(b.mean())) < FLAT_AMPLITUDE else 0.0
        return 0.0
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if den == 0.0 or not np.isfinite(den):
        return 0.0
    # Elementwise sum, NOT `am @ bm`: matmul dispatch hits spurious BLAS
    # warnings on some numpy builds (seen: 2.1.3/macOS arm) for identical math.
    return float(np.clip((am * bm).sum() / den, -1.0, 1.0))


def scaled_slope(x: np.ndarray, y: np.ndarray, *, scale: int | None = None) -> float:
    """Least-squares slope through ``(x, y)``, scaled by a profile length.

    The shared spelling of "net change from one end of a profile to the other":
    slope times ``scale`` (the number of samples by default), so the value
    stays in the profile's own units instead of shrinking as the profile grows.
    0.0 for fewer than two samples.

    ``scale`` is explicit for callers whose profile covers a fixed number of
    bins but whose fit may see only the occupied ones (radial gradient).
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    slope = np.polyfit(np.asarray(x, dtype=np.float64), y, 1)[0]
    return float(slope * (len(y) if scale is None else scale))
