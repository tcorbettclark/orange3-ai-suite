"""Decoding the caller's array: alpha, luminance and value range."""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.images.decode import (
    drop_alpha,
    to_luminance,
    to_unit_scale,
)


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
    with pytest.raises(ValueError, match="3 \\(RGB\\) or 4 \\(RGBA\\)"):
        to_luminance(np.zeros((4, 4, 1)))


def test_drop_alpha_leaves_shorter_arrays_alone():
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    gray = np.zeros((2, 2), dtype=np.float32)
    assert drop_alpha(rgb) is rgb
    assert drop_alpha(gray) is gray
    assert drop_alpha(np.zeros((2, 2, 4))).shape == (2, 2, 3)


def test_unit_scale_accepts_unit_and_8bit_ranges():
    assert np.allclose(to_unit_scale(np.full((2, 2), 0.5, np.float32)), 0.5)
    assert np.allclose(to_unit_scale(np.full((2, 2), 255, np.uint8)), 1.0)
    assert np.allclose(to_unit_scale(np.full((2, 2), 51, np.uint8)), 0.2)


def test_unit_scale_rejects_16bit_values():
    with pytest.raises(ValueError, match="must be in \\[0, 1\\] or \\[0, 255\\]"):
        to_unit_scale(np.full((2, 2), 60000, np.uint16))


def test_unit_scale_rejects_negative_values():
    # Neither range has a negative floor; passing one through would invert
    # every threshold that assumes intensity is non-negative.
    with pytest.raises(ValueError, match="got min -3"):
        to_unit_scale(np.array([[-3.0, 0.5]]))
    assert np.allclose(to_unit_scale(np.zeros((2, 2), np.float32)), 0.0)  # zero is fine


def test_unit_scale_passes_an_empty_image_through():
    empty = np.zeros((0, 0), np.float32)
    out = to_unit_scale(empty)
    assert out.shape == (0, 0)  # no min or max to interpret; nothing to raise about
