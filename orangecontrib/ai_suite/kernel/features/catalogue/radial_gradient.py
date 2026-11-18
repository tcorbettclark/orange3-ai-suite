import numpy as np

from .._conventions import EPS, scaled_slope
from .._imaging import mass_moments
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions

RADIAL_BINS = 8


class RadialGradient(Feature):
    name = "radial_gradient"
    description = Description(
        title="Centre-edge gradient",
        summary="Whether brightness grows or falls from the image's own centre of mass outward.",
        intuition=(
            "A glowing lamp, or a bright ball in the middle of the frame, is brightest at the "
            "centre and fades outward, so this scores negative. A ring of cloud around a "
            "clear centre, or a dark subject surrounded by bright light, scores positive. It "
            "measures the trend from the middle outward, not the exact shape of the falloff."
        ),
        technical=(
            "The image is binned into 8 rings by distance from its own centre of mass, the "
            "mean intensity of each occupied ring is taken, and a least-squares line is "
            "fitted against ring position. The reported value is the slope times 8, the net "
            "change from centre to edge. Empty rings are skipped; fewer than two occupied "
            "rings returns 0.0. Rings follow the mass, not the frame, so a cropped subject "
            "still reports its own gradient."
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        moments = mass_moments(image)
        h, w = image.shape
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
        radius = np.hypot(xx - moments.com_x, yy - moments.com_y)
        if radius.max() < EPS:
            return 0.0
        bins = np.clip((radius / radius.max() * RADIAL_BINS).astype(int), 0, RADIAL_BINS - 1)
        sums = np.bincount(bins.ravel(), weights=image.ravel(), minlength=RADIAL_BINS)
        counts = np.bincount(bins.ravel(), minlength=RADIAL_BINS)
        occupied = counts > 0
        if occupied.sum() < 2:
            return 0.0
        centres = (np.arange(RADIAL_BINS) + 0.5) / RADIAL_BINS
        means = sums[occupied] / counts[occupied]
        return scaled_slope(centres[occupied], means, scale=RADIAL_BINS)
