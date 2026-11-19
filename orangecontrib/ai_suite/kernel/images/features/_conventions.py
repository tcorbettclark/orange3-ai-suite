"""Shared numeric conventions: what "zero", "flat" and "net change" mean.

:data:`EPS` and :data:`FLAT_AMPLITUDE` are the thresholds; :func:`is_flat`,
:func:`safe_corr` and :func:`scaled_slope` apply them. Image measurements live in
:mod:`_measurements`.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12  # "numerically zero": division, range and empty-mass guards

FLAT_AMPLITUDE = 1e-6  # centred amplitude below this counts as a flat input

_FLAT_POLICIES = ("zero", "equal_is_one")


def is_flat(array: np.ndarray) -> bool:
    """Whether an array carries no structure, by centred amplitude."""
    values = np.asarray(array, dtype=np.float64).ravel()
    return float(np.abs(values - values.mean()).max()) < FLAT_AMPLITUDE


def safe_corr(a: np.ndarray, b: np.ndarray, *, flat_policy: str) -> float:
    """Pearson correlation with an explicit policy for flat (constant) inputs.

    A flat side has zero centred amplitude, so its correlation is undefined.
    ``flat_policy`` states what that means for the caller's feature:

    - ``"zero"``: a flat side carries no structure to agree or disagree about,
      so the result is 0.0.
    - ``"equal_is_one"``: both sides constant and equal is 1.0; one side
      constant, or both constant but unequal, is 0.0.
    """
    if flat_policy not in _FLAT_POLICIES:
        raise ValueError(f"unknown flat policy {flat_policy!r}; valid: {_FLAT_POLICIES}")
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    flat_a, flat_b = is_flat(a), is_flat(b)
    if flat_a or flat_b:
        if flat_policy == "equal_is_one" and flat_a and flat_b:
            return 1.0 if abs(float(a.mean()) - float(b.mean())) < FLAT_AMPLITUDE else 0.0
        return 0.0
    am = a - a.mean()
    bm = b - b.mean()
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if not np.isfinite(den):  # only non-finite input reaches here; flat is above
        return 0.0
    # Elementwise sum, NOT `am @ bm`: matmul dispatch hits spurious BLAS
    # warnings on some numpy builds (seen: 2.1.3/macOS arm) for identical math.
    return float(np.clip((am * bm).sum() / den, -1.0, 1.0))


def scaled_slope(x: np.ndarray, y: np.ndarray, *, scale: int | None = None) -> float:
    """Least-squares slope through ``(x, y)``, scaled by a profile length.

    The result is the slope times ``scale``, by default the number of samples, so
    it stays in the profile's own units. 0.0 for fewer than two samples.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    slope = np.polyfit(np.asarray(x, dtype=np.float64), y, 1)[0]
    return float(slope * (len(y) if scale is None else scale))
