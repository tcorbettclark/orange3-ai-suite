"""Stability tests — a first-class requirement of the domain spec (§2, §5):

"small affine jitter and varying denoising strength must not flip feature
categories (e.g. 'two peaks' -> 'one peak'); if they do, the feature is
noise."

Category features (peak/hole/median-crossing counts) must be *identical*
under 1-pixel shifts and small additive noise; continuous features must stay
within tolerance.
"""

import numpy as np
import pytest

from nn_kernel.features import FeatureOptions, extract_features
from synth import cloud_image, crop_shift, digit8_image, ring_image, shift_image

COUNT_FEATURES = frozenset({
    "row_profile_peaks",
    "column_profile_peaks",
    "diagonal_profile_peaks",
    "anti_diagonal_profile_peaks",
    "median_upcrossing_count",
    "hole_count",
})
TOL_CONTINUOUS = 0.05  # relative to max(1, |value|)
TOL_NOISY = 0.20  # additive noise genuinely adds fine texture; looser bar


def _assert_stable(base: dict, other: dict, tol=TOL_CONTINUOUS):
    for name in base:
        a, b = base[name], other[name]
        if name in COUNT_FEATURES:
            assert a == b, f"{name} flipped: {a} -> {b} (category feature)"
        else:
            assert abs(a - b) <= tol * max(1.0, abs(a)), (
                f"{name} moved {a:.4f} -> {b:.4f}"
            )


@pytest.fixture(scope="module")
def cloud_base():
    return extract_features(cloud_image(seed=1)).by_group()["D"]


def test_one_pixel_shift_texture(cloud_base):
    img = cloud_image(seed=1)
    # Translation by cropping: no synthetic border is introduced (a zero-fill
    # shift draws a fake edge that legitimately changes texture features).
    for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        shifted = extract_features(crop_shift(img, dy, dx), groups="D")
        _assert_stable(cloud_base, shifted.by_group()["D"])


def test_small_noise_jitter_texture(cloud_base):
    img = cloud_image(seed=1)
    rng = np.random.default_rng(42)
    for _ in range(3):
        jittered = np.clip(
            img + rng.normal(0.0, 0.02, size=img.shape).astype(np.float32),
            0.0, 1.0,
        )
        _assert_stable(
            cloud_base,
            extract_features(jittered, groups="D").by_group()["D"],
            tol=TOL_NOISY,
        )


def test_hole_count_stable_under_shift():
    digit = digit8_image()
    assert extract_features(digit)["hole_count"] == 2.0
    for dy, dx in ((1, 1), (-1, 0), (0, 2), (5, 0), (0, -5)):
        shifted = extract_features(shift_image(digit, dy, dx))
        assert shifted["hole_count"] == 2.0


def test_ring_hole_count_stable_under_shift():
    ring = ring_image()
    for dy, dx in ((1, 0), (0, 1), (-1, -1)):
        assert extract_features(shift_image(ring, dy, dx))["hole_count"] == 1.0


def test_denoising_strength_does_not_flip_categories():
    img = cloud_image(seed=2)
    base = extract_features(img).by_group()["D"]
    for ksize in (3, 5):
        opts = FeatureOptions(denoise=True, denoise_ksize=ksize)
        f = extract_features(img, options=opts, groups="D").by_group()["D"]
        # denoising a smooth image is near-idempotent; counts must not flip
        for name in COUNT_FEATURES & set(base):
            assert f[name] == base[name], (name, base[name], f[name])


def test_working_size_change_keeps_digit_categories():
    digit = digit8_image()
    for size in (64, 128, 256):
        vec = extract_features(digit, options=FeatureOptions(working_size=size))
        assert vec["hole_count"] == 2.0


def test_shift_is_true_translation_not_wrap():
    img = np.zeros((8, 8), dtype=np.float32)
    img[2, 2] = 1.0
    moved = shift_image(img, 1, 1)
    assert moved[3, 3] == 1.0 and moved[2, 2] == 0.0
    assert moved[0, 0] == 0.0  # no wrap-around