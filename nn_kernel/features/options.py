"""Configuration for feature extraction and preprocessing."""

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
        features are comparable across images (agreed 2025, spec Q4).
        ``None`` keeps native resolution.
    centre_crop:
        Crop to the central square before resizing.
    contrast_normalise:
        Robust percentile normalisation to [0, 1] (p1-p99 clip-and-scale).
    denoise:
        Median-filter denoising before normalisation. The strength is an
        explicit scale parameter (spec §2: denoising must not silently erase
        signal) — the kernel never denoises unless asked.
    denoise_ksize:
        Median filter kernel size, clamped to {3, 5} (float-image limit).
    peak_prominence_frac:
        Prominence threshold for all peak-count features, as a fraction of
        the profile's dynamic range (agreed Q1, default 0.10).
    edge_density_frac:
        Edge-density threshold as a fraction of the image's own gradient
        p99 (see features/README.md on why a fixed percentile would be
        degenerate).
    """

    working_size: int | None = 128
    centre_crop: bool = True
    contrast_normalise: bool = True
    denoise: bool = False
    denoise_ksize: int = 3
    peak_prominence_frac: float = 0.10
    edge_density_frac: float = 0.5