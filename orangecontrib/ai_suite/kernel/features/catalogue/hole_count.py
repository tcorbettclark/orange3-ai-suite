"""Enclosed background regions: the most interpretable digit feature there is.

Foreground is brighter-than-Otsu-threshold, and holes are background
components that do not touch the image border. "8" -> 2, "0" -> 1, "1" -> 0.
Polarity is fixed and documented: a genuinely dark-object-on-light image has
no holes by this definition and must be inverted before extraction.
"""

import cv2
import numpy as np

from .._primitives import otsu_foreground
from ..feature import Feature, FeatureOptions


class HoleCount(Feature):
    name = "hole_count"
    definition = (
        "Enclosed background regions (bright-foreground Otsu binarisation): "
        "'8' -> 2, '0' -> 1, '1' -> 0 — the most interpretable digit feature."
    )
    is_count = True

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        background = (~otsu_foreground(img)).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(background, connectivity=4)
        border = {
            int(label)
            for label in np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        }
        return float(sum(1 for label in range(1, n_labels) if label not in border))
