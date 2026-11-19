"""Configuration for image preprocessing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

_MEDIAN_KERNELS = (3, 5)


@dataclass(frozen=True)
class FeatureOptions:
    """Options shared by preprocessing and feature computation.

    Invalid values raise ``ValueError`` on construction.

    Attributes
    ----------
    working_size:
        Images are centre-cropped to a square and resized to
        ``working_size x working_size`` before feature computation.
        ``None`` keeps native resolution - and then the pixel-scale features
        (``fine_texture``, ``coarse_texture``, ``autocorrelation_half_width``)
        change meaning with the image size, because their windows are in pixels.
    centre_crop:
        Crop to the central square before resizing.
    contrast_normalise:
        Robust percentile normalisation to [0, 1] (p1-p99 clip-and-scale).
        With it off, features read the input's own intensity scale.
    denoise:
        Median-filter denoising before normalisation, off by default.
    denoise_ksize:
        Median filter kernel size: 3 or 5. A median filter needs an odd kernel,
        and 1 is a no-op.
    peak_prominence_frac:
        Prominence threshold for all peak-count features, as a fraction of
        the profile's dynamic range. 0 counts every local maximum; 1 counts
        none.
    edge_density_frac:
        Edge-density threshold as a fraction of the image's own gradient p99.
    """

    working_size: int | None = 128
    centre_crop: bool = True
    contrast_normalise: bool = True
    denoise: bool = False
    denoise_ksize: int = 3
    peak_prominence_frac: float = 0.10
    edge_density_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.working_size is not None and (
            isinstance(self.working_size, bool)
            or not isinstance(self.working_size, int)
            or self.working_size < 1
        ):
            raise ValueError(
                f"working_size must be a positive integer or None, got {self.working_size!r}"
            )
        for flag in ("centre_crop", "contrast_normalise", "denoise"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
        if self.denoise_ksize not in _MEDIAN_KERNELS:
            raise ValueError(
                f"denoise_ksize must be one of {_MEDIAN_KERNELS}, got {self.denoise_ksize!r}"
            )
        for name in ("peak_prominence_frac", "edge_density_frac"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a fraction in [0, 1], got {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1], got {value!r}")
