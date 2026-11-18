"""Preprocessing pipeline tests."""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.images.features import FeatureOptions
from orangecontrib.ai_suite.kernel.images.features._preprocessing import (
    preprocess,
    preprocess_colour,
)
from synth import uniform_image


def test_preprocess_colour_returns_unit_range_from_8bit_input():
    rgb = np.full((8, 8, 3), 128, np.uint8)
    out = preprocess_colour(rgb, FeatureOptions(working_size=None))
    assert out.max() <= 1.0 and np.isclose(out.max(), 128 / 255)


def test_preprocess_colour_returns_unit_range_from_grayscale_too():
    gray = np.full((8, 8), 128, np.uint8)
    out = preprocess_colour(gray, FeatureOptions(working_size=None))
    assert np.isclose(out.max(), 128 / 255)


def test_preprocess_without_normalisation_keeps_the_input_scale():
    rgb = np.full((8, 8, 3), 200, np.uint8)
    opts = FeatureOptions(working_size=None, contrast_normalise=False)
    assert np.isclose(preprocess(rgb, opts).max(), 200.0)  # grey 200 keeps its scale
    assert preprocess(rgb, FeatureOptions(working_size=None)).max() <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"working_size": 0},
        {"working_size": -8},
        {"denoise_ksize": 4},
        {"denoise_ksize": 1},
        {"denoise_ksize": 7},
        {"peak_prominence_frac": -0.1},
        {"peak_prominence_frac": 1.5},
        {"edge_density_frac": -1.0},
        {"centre_crop": "yes"},
    ],
)
def test_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FeatureOptions(**kwargs)


def test_preprocess_squares_and_resizes():
    img = np.zeros((100, 200), dtype=np.float32)
    out = preprocess(img, FeatureOptions(working_size=64))
    assert out.shape == (64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_native_when_working_size_none():
    img = np.zeros((100, 200), dtype=np.float32)
    opts = FeatureOptions(working_size=None, centre_crop=False, contrast_normalise=False)
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
        img[y, x] = 1.0  # p99 clip keeps them bright
    noisy = preprocess(img, FeatureOptions(working_size=None))
    clean = preprocess(img, FeatureOptions(working_size=None, denoise=True))
    assert noisy.max() > 0.9  # spikes survive normalisation
    assert clean.max() < 0.1  # median filter removes isolated spikes
