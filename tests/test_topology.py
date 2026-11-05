"""Group E — topology."""

import numpy as np

from orangecontrib.ai_suite.kernel.features import FeatureOptions, topology
from synth import digit8_image, ring_image, vertical_bar_image

OPTS = FeatureOptions()


def _compute(img):
    return topology.compute(img, OPTS)


def test_hole_counts_digit_like():
    assert _compute(digit8_image())["hole_count"] == 2.0
    assert _compute(ring_image())["hole_count"] == 1.0
    assert _compute(vertical_bar_image())["hole_count"] == 0.0


def test_hole_count_touching_border_is_not_a_hole():
    img = np.zeros((128, 128), dtype=np.float32)
    img[10:118, 10:118] = 1.0
    img[40:88, 40:88] = 0.0  # enclosed -> 1 hole
    assert _compute(img)["hole_count"] == 1.0

    img2 = np.zeros((128, 128), dtype=np.float32)
    img2[10:118, 10:118] = 1.0
    img2[40:110, 40:88] = 0.0  # enclosed cut, well inside the bright block
    assert _compute(img2)["hole_count"] == 1.0

    img3 = np.zeros((128, 128), dtype=np.float32)
    img3[10:118, 10:118] = 1.0
    img3[40:118, 40:88] = 0.0  # cut reaches the block's bottom edge (row 117
    # adjoins the dark background at row 118) -> connected to the outside
    assert _compute(img3)["hole_count"] == 0.0


def test_foreground_area_fraction_ring():
    f = _compute(ring_image())
    expected = np.pi * (0.38**2 - 0.22**2)  # annulus fraction of the square
    assert abs(f["foreground_area_fraction"] - expected) < 0.02


def test_stroke_crossings_vertical_bar():
    # Each row enters and leaves the (bright, minority) bar: 2 crossings.
    f = _compute(vertical_bar_image())
    assert abs(f["stroke_crossings_median_level"] - 2.0) < 1e-6


def test_flat_image_guards():
    img = np.zeros((128, 128), dtype=np.float32)
    f = _compute(img)
    assert f["hole_count"] == 0.0
    assert f["foreground_area_fraction"] == 0.0
    assert f["stroke_crossings_median_level"] == 0.0
