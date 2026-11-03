"""Synthetic test images — no external data required."""

from __future__ import annotations

import cv2
import numpy as np


def uniform_image(level: float = 0.5, size: int = 128) -> np.ndarray:
    return np.full((size, size), level, dtype=np.float32)


def band_image(
    bands: tuple[tuple[int, int], ...],
    size: int = 128,
    background: float = 0.0,
    level: float = 1.0,
) -> np.ndarray:
    """Horizontal bright bands as (start_row, end_row) ranges."""
    img = np.full((size, size), background, dtype=np.float32)
    for start, end in bands:
        img[start:end, :] = level
    return img


def vertical_bar_image(size: int = 128, width: int = 12) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    img[:, c - width // 2 : c + width // 2] = 1.0
    return img


def ring_image(
    size: int = 128,
    centre: tuple[float, float] = (0.5, 0.5),
    outer: float = 0.38,
    inner: float = 0.22,
) -> np.ndarray:
    """An annulus (digit "0"-like). Centre fractions map onto the pixel
    grid, so 0.5 falls on the half-pixel mirror axis for even sizes."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = centre[0] * (size - 1), centre[1] * (size - 1)
    r = np.hypot(yy - cy, xx - cx)
    return ((r <= outer * size) & (r >= inner * size)).astype(np.float32)


def digit8_image(size: int = 128) -> np.ndarray:
    """Two stacked annuli -> hole count 2."""
    top = ring_image(size, centre=(0.32, 0.5), outer=0.16, inner=0.08)
    bottom = ring_image(size, centre=(0.68, 0.5), outer=0.16, inner=0.08)
    return np.maximum(top, bottom)


def slope_image(size: int = 128, horizontal: bool = False) -> np.ndarray:
    """Brightness ramp: bright at the larger-index edge."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ramp = axis[None, :] if horizontal else axis[:, None]
    return np.broadcast_to(ramp, (size, size)).copy()


def ellipse_blob_image(
    size: int = 128,
    centre: tuple[float, float] = (0.5, 0.5),
    ry: float = 0.15,
    rx: float = 0.35,
    level: float = 1.0,
) -> np.ndarray:
    """A wide bright ellipse blob (streak) on dark background."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = centre[0] * (size - 1), centre[1] * (size - 1)
    r2 = ((yy - cy) / (ry * size)) ** 2 + ((xx - cx) / (rx * size)) ** 2
    return (r2 <= 1.0).astype(np.float32) * level


def cloud_image(seed: int = 0, size: int = 128) -> np.ndarray:
    """Smooth random blob (blurred noise clipped and rescaled) — cloud-like."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=size / 8)
    smooth = np.clip(smooth, None, 0.0)
    lo = smooth.min()
    return ((smooth - lo) / (smooth.max() - lo + 1e-12)).astype(np.float32)


def shift_image(img: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """Translate an image by (dy, dx), filling with `fill` (true shift, no wrap).

    Note: zero fill creates a synthetic border (which genuinely changes
    texture features); use `crop_shift` for translation-stability tests of
    texture.
    """
    out = np.full_like(img, fill)
    h, w = img.shape
    ys, xs = max(dy, 0), max(dx, 0)
    ye, xe = h + min(dy, 0), w + min(dx, 0)
    out[ys:ye, xs:xe] = img[: ye - ys, : xe - xs]
    return out


def crop_shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translation by cropping: remove rows/cols from opposite edges, so the
    content moves without introducing any synthetic border."""
    h, w = img.shape
    return img[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)].copy()
