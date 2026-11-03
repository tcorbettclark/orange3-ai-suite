"""Preprocessing pipeline tests."""

import numpy as np
import pytest

from nn_kernel.features import FeatureOptions, preprocess, to_luminance
from synth import uniform_image


def test_luminance_rgb_weights():
    red = np.zeros((4, 4, 3), dtype=np.float32)
    red[..., 0] = 255.0
    lum = to_luminance(red)
    assert lum.dtype == np.float32
    assert np.allclose(lum, 255.0 * 0.299)


def test_luminance_alpha_ignored():
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., 0] = 100.0
    rgba[..., 3] = 255.0  # alpha must be dropped, not averaged in
    assert np.allclose(to_luminance(rgba), 100.0 * 0.299)


def test_luminance_grayscale_passthrough():
    gray = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(to_luminance(gray), gray)


def test_luminance_rejects_bad_channels():
    with pytest.raises(ValueError):
        to_luminance(np.zeros((4, 4, 2)))


def test_preprocess_squares_and_resizes():
    img = np.zeros((100, 200), dtype=np.float32)
    out = preprocess(img, FeatureOptions(working_size=64))
    assert out.shape == (64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_native_when_working_size_none():
    img = np.zeros((100, 200), dtype=np.float32)
    opts = FeatureOptions(working_size=None, centre_crop=False,
                          contrast_normalise=False)
    assert preprocess(img, opts).shape == (100, 200)


def test_preprocess_flat_image_is_zeroed():
    out = preprocess(uniform_image(0.5), FeatureOptions(working_size=None))
    assert np.allclose(out, 0.0)


def test_preprocess_contrast_normalise_uses_full_range():
    from synth import slope_image
    out = preprocess(slope_image(128), FeatureOptions(working_size=None))
    assert out.min() <= 0.05
    assert out.max() >= 0.95


def test_preprocess_denoise_removes_isolated_spikes():
    img = np.zeros((128, 128), dtype=np.float32)
    rng = np.random.default_rng(0)
    for y, x in rng.integers(5, 123, size=(200, 2)):  # >1% of pixels so the
        img[y, x] = 1.0                               # p99 clip keeps them bright
    noisy = preprocess(img, FeatureOptions(working_size=None))
    clean = preprocess(img, FeatureOptions(working_size=None, denoise=True))
    assert noisy.max() > 0.9  # spikes survive normalisation
    assert clean.max() < 0.1  # median filter removes isolated spikes