"""Configuration for image preprocessing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureOptions:
    """Options shared by preprocessing and feature computation.

    Attributes
    ----------
    working_size:
        Images are centre-cropped to a square and resized to
        ``working_size x working_size`` before feature computation, so
        features are comparable across images. ``None`` keeps native
        resolution.
    centre_crop:
        Crop to the central square before resizing.
    contrast_normalise:
        Robust percentile normalisation to [0, 1] (p1-p99 clip-and-scale).
    denoise:
        Median-filter denoising before normalisation. Denoising is off by
        default and its strength is explicit: it must never silently erase the
        signal being measured.
    denoise_ksize:
        Median filter kernel size. It must be odd and at least 3: 1 is a
        no-op and even kernels are invalid for a median filter. Values are
        clamped to 3-5, with even values rounded up (4 becomes 5).
    peak_prominence_frac:
        Prominence threshold for all peak-count features, as a fraction of
        the profile's dynamic range.
    edge_density_frac:
        Edge-density threshold as a fraction of the image's own gradient p99.
        A fixed percentile would return the same density for every image.
    """

    working_size: int | None = 128
    centre_crop: bool = True
    contrast_normalise: bool = True
    denoise: bool = False
    denoise_ksize: int = 3
    peak_prominence_frac: float = 0.10
    edge_density_frac: float = 0.5
