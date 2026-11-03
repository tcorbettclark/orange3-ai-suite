"""Group C: spatial mass distribution (6 features), where the bright matter sits.

All mass statistics are intensity-weighted and normalised by image size, so
they live in [0, 1] and are comparable across images.

Definitions
-----------
- centre of mass x/y: intensity-weighted mean coordinate, normalised to [0, 1]
  (0.5 = centred).
- horizontal/vertical spread: intensity-weighted standard deviation along each
  axis, divided by the image side (0 = all mass in one line, ~0.29 = uniform).
- elongation: eigenvalue ratio of the 2x2 intensity second-moment (covariance)
  matrix; >1 = streak along the major axis, ~1 = blob. 1.0 if degenerate.
- radial gradient: the preprocessed image is binned into 8 rings around its
  own centre of mass; the returned value is the linear-fit slope of mean
  intensity against ring position, scaled by the number of bins = net change
  from centre to edge. Positive = edges brighter, negative = centre-bright.
  Empty rings (centre of mass near a corner) are excluded from the fit; 0.0
  if fewer than two rings contain pixels.
"""

from __future__ import annotations

import numpy as np

from .options import FeatureOptions

_EPS = 1e-12
_RADIAL_BINS = 8

FEATURE_NAMES = (
    "centre_of_mass_x",
    "centre_of_mass_y",
    "horizontal_spread",
    "vertical_spread",
    "elongation",
    "radial_gradient",
)

DEFINITIONS = {
    "centre_of_mass_x": ("Left/right placement of the bright mass in [0, 1]; 0.5 = centred."),
    "centre_of_mass_y": ("Top/bottom placement of the bright mass in [0, 1]; 0.5 = centred."),
    "horizontal_spread": (
        "Wide vs. narrow bright mass: intensity-weighted std along x, divided by image width."
    ),
    "vertical_spread": (
        "Tall vs. squat bright mass: intensity-weighted std along y, divided by image height."
    ),
    "elongation": (
        "Streak (>1) vs. blob (~1): eigenvalue ratio of the intensity second-moment matrix."
    ),
    "radial_gradient": (
        "Mass concentrated at the centre (negative) vs. at the edges (positive): "
        "net intensity change from the image's own centre of mass outward "
        "over 8 rings."
    ),
}


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    h, w = img.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    col_mass = img.sum(axis=0).astype(np.float64)
    row_mass = img.sum(axis=1).astype(np.float64)
    total = float(img.sum())

    if total < _EPS:
        return {
            "centre_of_mass_x": 0.5,
            "centre_of_mass_y": 0.5,
            "horizontal_spread": 0.0,
            "vertical_spread": 0.0,
            "elongation": 1.0,
            "radial_gradient": 0.0,
        }

    comx = float((col_mass * xs).sum() / total)
    comy = float((row_mass * ys).sum() / total)

    varx = float((col_mass * (xs - comx) ** 2).sum() / total)
    vary = float((row_mass * (ys - comy) ** 2).sum() / total)
    # Cross term needs the full weighted expectation E[(x-cx)(y-cy)].
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cxy = float((img.astype(np.float64) * (xx - comx) * (yy - comy)).sum() / total)

    eigvals = np.linalg.eigvalsh(np.array([[varx, cxy], [cxy, vary]]))
    lo = float(eigvals[0])
    elong = float(eigvals[1] / lo) if lo > _EPS else 1.0

    return {
        "centre_of_mass_x": comx,
        "centre_of_mass_y": comy,
        "horizontal_spread": float(np.sqrt(max(varx, 0.0))),
        "vertical_spread": float(np.sqrt(max(vary, 0.0))),
        "elongation": elong,
        "radial_gradient": _radial_gradient(img, comx, comy),
    }


def _radial_gradient(img: np.ndarray, comx: float, comy: float) -> float:
    h, w = img.shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    r = np.hypot(xx - comx, yy - comy)
    if r.max() < _EPS:
        return 0.0
    bins = np.clip((r / r.max() * _RADIAL_BINS).astype(int), 0, _RADIAL_BINS - 1)
    sums = np.bincount(bins.ravel(), weights=img.ravel(), minlength=_RADIAL_BINS)
    counts = np.bincount(bins.ravel(), minlength=_RADIAL_BINS)
    occupied = counts > 0
    if occupied.sum() < 2:
        return 0.0
    centres = (np.arange(_RADIAL_BINS) + 0.5) / _RADIAL_BINS
    prof = sums[occupied] / counts[occupied]
    slope = np.polyfit(centres[occupied], prof, 1)[0]
    return float(slope * _RADIAL_BINS)
