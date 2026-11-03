"""Group E: topology (3 features), the classic digit features.

Definitions
-----------
Binarisation is adaptive everywhere: Otsu's threshold on the preprocessed
image, with **brighter-than-threshold = foreground** (bright object on dark
background is the convention for both digits and clouds). Polarity is fixed
and documented; a genuinely dark-object-on-light image can be inverted by
the caller before extraction.

- hole count: connected components of the background that do not touch the
  image border. "8" -> 2, "0" -> 1, "1" -> 0 — the most interpretable digit
  feature there is.
- foreground area fraction: fraction of pixels above the Otsu threshold
  (ink / cloud coverage).
- stroke crossings: per row, the number of times the row crosses the image's
  global median intensity level (transitions between below/above); averaged
  over rows. Dense version of the classic OCR crossing-count feature; open
  strokes ("1", "7") score low, dense strokes ("8") score high.
"""

from __future__ import annotations

import cv2
import numpy as np

from .options import FeatureOptions

_EPS = 1e-12

FEATURE_NAMES = (
    "hole_count",
    "foreground_area_fraction",
    "stroke_crossings_median_level",
)

DEFINITIONS = {
    "hole_count": (
        "Enclosed background regions (bright-foreground Otsu binarisation): "
        "'8' -> 2, '0' -> 1, '1' -> 0 — the most interpretable digit feature."
    ),
    "foreground_area_fraction": (
        "Ink / cloud coverage: fraction of pixels above the Otsu threshold."
    ),
    "stroke_crossings_median_level": (
        "Dense version of the classic OCR stroke-crossing count: crossings of "
        "the global median level per row, averaged; open strokes ('1', '7') "
        "score low, dense strokes ('8') high."
    ),
}


def _otsu_foreground(img: np.ndarray) -> np.ndarray:
    """Bright-foreground Otsu binarisation.

    cv2's THRESH_OTSU accepts only 8U/16U input, so the image is min-max
    scaled to uint8 and the resulting threshold mapped back to the original
    scale (Otsu is invariant under affine rescaling, so this is exact).
    A flat image has no foreground.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < _EPS:
        return np.zeros(img.shape, dtype=bool)
    u8 = np.round((img - lo) / (hi - lo) * 255.0).astype(np.uint8)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img > lo + float(threshold) / 255.0 * (hi - lo)


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    fg = _otsu_foreground(img)

    bg = (~fg).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(bg, connectivity=4)
    border_labels = {
        int(v) for v in np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    }
    holes = sum(1 for label in range(1, n_labels) if label not in border_labels)

    below = img <= np.median(img)  # below-or-equal vs. above: robust when the
    # median coincides with the background level (minority bright object)
    transitions = (below[:, 1:] != below[:, :-1]).sum(axis=1)

    return {
        "hole_count": float(holes),
        "foreground_area_fraction": float(fg.mean()),
        "stroke_crossings_median_level": float(transitions.mean()),
    }
