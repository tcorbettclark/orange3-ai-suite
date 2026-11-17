"""Typical feature size: the lag at which the image stops resembling itself.

The 2-D autocorrelation is FFT-based on the mean-subtracted image, normalised
by its central value (the variance), then radially averaged. The returned lag
is the first radius where that profile drops below 0.5, linearly interpolated;
if it never drops, the maximum radius is returned. A flat image has no
structure and returns 0.0.
"""

import numpy as np

from .._stats import EPS
from ..feature import Feature, FeatureOptions


class AutocorrelationHalfWidth(Feature):
    name = "autocorrelation_half_width"
    definition = (
        "Typical feature size in pixels: first lag where the normalised "
        "autocorrelation drops below 0.5."
    )

    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        h, w = img.shape
        centred = img.astype(np.float64) - img.mean()
        if centred.std() < EPS:
            return 0.0
        power = np.abs(np.fft.fft2(centred)) ** 2
        autocorrelation = np.fft.fftshift(np.fft.ifft2(power).real)
        autocorrelation /= autocorrelation[h // 2, w // 2]  # normalise by lag 0
        yy, xx = np.ogrid[:h, :w]
        radius = np.hypot(yy - h // 2, xx - w // 2).round().astype(int)
        sums = np.bincount(radius.ravel(), weights=autocorrelation.ravel())
        counts = np.bincount(radius.ravel())
        profile = sums / np.maximum(counts, 1)
        rmax = len(profile) - 1
        for r in range(1, rmax + 1):
            if profile[r] < 0.5:
                prev, cur = profile[r - 1], profile[r]
                frac = 0.0 if cur >= prev else (prev - 0.5) / (prev - cur)
                return float(r - 1 + frac)
        return float(rmax)
