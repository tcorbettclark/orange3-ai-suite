"""Concentrated in the middle, or spread to the edges.

Ring means are taken around the image's own centre of mass, not the frame
centre, so the gradient describes the mass rather than the crop. The fit runs
over occupied rings only and is scaled by the full ring count, so an empty ring
near a corner does not change the unit.
"""

import numpy as np

from .._primitives import mass_moments
from .._stats import EPS, scaled_slope
from ..feature import Feature, FeatureOptions

RADIAL_BINS = 8


class RadialGradient(Feature):
    name = "radial_gradient"
    definition = (
        "Mass concentrated at the centre (negative) vs. at the edges (positive): "
        "net intensity change from the image's own centre of mass outward "
        "over 8 rings."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        moments = mass_moments(img)
        h, w = img.shape
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
        radius = np.hypot(xx - moments.com_x, yy - moments.com_y)
        if radius.max() < EPS:
            return 0.0
        bins = np.clip((radius / radius.max() * RADIAL_BINS).astype(int), 0, RADIAL_BINS - 1)
        sums = np.bincount(bins.ravel(), weights=img.ravel(), minlength=RADIAL_BINS)
        counts = np.bincount(bins.ravel(), minlength=RADIAL_BINS)
        occupied = counts > 0
        if occupied.sum() < 2:
            return 0.0
        centres = (np.arange(RADIAL_BINS) + 0.5) / RADIAL_BINS
        means = sums[occupied] / counts[occupied]
        return scaled_slope(centres[occupied], means, scale=RADIAL_BINS)
