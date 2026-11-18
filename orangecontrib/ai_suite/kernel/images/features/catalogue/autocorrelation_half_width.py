import numpy as np

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class AutocorrelationHalfWidth(Feature):
    name = "autocorrelation_half_width"
    description = Description(
        title="Feature size",
        summary="The lag, in pixels, at which the image stops resembling itself.",
        intuition=(
            "Squint and slide the photo across itself: how far can you move it before it "
            "stops matching? A broad cloud or a wide stripe stays matched for tens of pixels, "
            "so this scores high; fine grass or gravel loses the match after two or three, so "
            "it scores low. It is a typical size of the pattern, not the size of any "
            "particular object."
        ),
        technical=(
            "The 2-D autocorrelation is computed from the FFT of the mean-subtracted image, "
            "normalised by its lag-zero value and averaged over rounded radii. The reported "
            "lag is the first radius where that profile falls below 0.5, linearly "
            "interpolated between samples; if it never falls, the maximum radius is returned. "
            "A flat image returns 0.0. Pixels, so it scales with working size."
        ),
        references=(
            "[Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation)",
            "[NumPy FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        h, w = image.shape
        if is_flat(image):
            return 0.0
        centred = image.astype(np.float64) - image.mean()
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
