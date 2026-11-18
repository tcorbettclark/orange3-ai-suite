import cv2
import numpy as np

from .._measurements import otsu_foreground
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class HoleCount(Feature):
    name = "hole_count"
    description = Description(
        title="Hole count",
        summary="Number of enclosed dark regions surrounded by bright pixels.",
        intuition=(
            "Think of the loops in 8, 6, 0 and 9 - each one is a hole. A drawing of an 8 "
            "scores 2, a 0 scores 1, a 1 scores 0. It also counts the windows in a photo of a "
            "building and the gaps between bubbles in foam. It stays at 0 for anything solid "
            "or open: a filled circle, a single stroke, grass, or cloud, where nothing is "
            "completely surrounded."
        ),
        technical=(
            "Otsu's method thresholds the preprocessed luminance image and the brighter class "
            "is taken as foreground. Holes are 4-connected background components that do not "
            "touch the image border, counted with cv2.connectedComponents. Polarity is fixed, "
            "so a dark object on a light background has no holes unless it is inverted first. "
            "A whole number, and exactly stable under small shifts for an unambiguous shape."
        ),
        references=(
            "[Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method)",
            "[OpenCV: connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        background = (~otsu_foreground(image)).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(background, connectivity=4)
        border = {
            int(label)
            for label in np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        }
        return float(sum(1 for label in range(1, n_labels) if label not in border))
