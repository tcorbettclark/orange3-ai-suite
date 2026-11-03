"""Group G: colour (4 features). Optional and off by default.

Colour is the deliberate shortcut. ``bluish_fraction`` alone can solve "is
there sky in this photo?", which teaches nothing about spatial structure.
Groups A-F therefore run on luminance only; switching this group on is the
shortcut-learning demonstration.

These features read the colour image (raw RGB in [0, 1]); the pipeline's
luminance image is untouched. The definitions are deliberately naive, and
that naivety is the point:

- bluish_fraction: fraction of pixels with B > R + 0.04 - sky presence.
- blueness_mean: mean(B - R), the overall sky colour balance.
- saturation_mean: mean(max-min over channels) - colourfulness.
- greenery_fraction: fraction with G > R and G > B - terrain, trees and other
  ground clutter.

Grayscale input has no colour: all four features are 0.0 by definition.
"""

from __future__ import annotations

import numpy as np

from .options import FeatureOptions

FEATURE_NAMES = (
    "bluish_fraction",
    "blueness_mean",
    "saturation_mean",
    "greenery_fraction",
)

DEFINITIONS = {
    "bluish_fraction": (
        "The shortcut: fraction of pixels bluer than red + 0.04 - sky "
        "presence, which answers 'is there sky in this photo' trivially. "
        "Switching this group on demonstrates shortcut learning."
    ),
    "blueness_mean": ("Overall sky colour balance: mean(B - R) in [-1, 1]."),
    "saturation_mean": ("Colourfulness: mean of (max - min) channel intensity."),
    "greenery_fraction": (
        "Ground-clutter detector: fraction of pixels greener than both red "
        "and blue (grass, trees, terrain)."
    ),
}


def compute(colour: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    """colour: HxWx3 float in [0, 1] (RGB) or a grayscale array (-> zeros)."""
    if colour.ndim == 2:
        return {name: 0.0 for name in FEATURE_NAMES}
    r = colour[..., 0].astype(np.float64)
    g = colour[..., 1].astype(np.float64)
    b = colour[..., 2].astype(np.float64)
    return {
        "bluish_fraction": float((b > r + 0.04).mean()),
        "blueness_mean": float((b - r).mean()),
        "saturation_mean": float((colour.max(axis=-1) - colour.min(axis=-1)).mean()),
        "greenery_fraction": float(((g > r) & (g > b)).mean()),
    }
