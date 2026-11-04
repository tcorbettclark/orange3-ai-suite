"""Raw pixel flattening: names, shapes, scaling and dimensions."""

import numpy as np
import pytest

from nn_kernel.pixels import pixel_feature_names, pixel_features


def test_pixel_feature_names_are_row_major():
    names = pixel_feature_names(3)
    assert names == (
        "pixel_r0_c0",
        "pixel_r0_c1",
        "pixel_r0_c2",
        "pixel_r1_c0",
        "pixel_r1_c1",
        "pixel_r1_c2",
        "pixel_r2_c0",
        "pixel_r2_c1",
        "pixel_r2_c2",
    )
    assert len(pixel_feature_names(10)) == 100


def test_pixel_features_shape_and_range():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 255
    out = pixel_features(image, 4)
    assert out.shape == (16,)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() == pytest.approx(1.0)


def test_pixel_features_use_luminance():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[..., 0] = 255  # red
    out = pixel_features(image, 1)
    assert out[0] == pytest.approx(0.299, abs=1e-3)


def test_pixel_features_accept_float_zero_to_one():
    image = np.full((4, 4), 0.5, dtype=np.float32)
    out = pixel_features(image, 2)
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_pixel_features_size_must_be_positive():
    with pytest.raises(ValueError):
        pixel_feature_names(0)
    with pytest.raises(ValueError):
        pixel_features(np.zeros((4, 4)), 0)
