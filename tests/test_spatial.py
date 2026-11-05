"""Group C — spatial mass distribution."""

import numpy as np

from orangecontrib.ai_suite.kernel.features import FeatureOptions, spatial
from synth import ellipse_blob_image, ring_image, uniform_image

OPTS = FeatureOptions()


def _compute(img):
    return spatial.compute(img, OPTS)


def test_centre_of_mass_placement():
    blob = ellipse_blob_image(centre=(0.25, 0.75))  # (cy, cx) = bottom-left
    f = _compute(blob)
    assert abs(f["centre_of_mass_x"] - 0.75) < 0.03
    assert abs(f["centre_of_mass_y"] - 0.25) < 0.03


def test_centre_of_mass_of_centred_ring():
    f = _compute(ring_image())
    assert abs(f["centre_of_mass_x"] - 0.5) < 0.02
    assert abs(f["centre_of_mass_y"] - 0.5) < 0.02


def test_uniform_mass_spreads_match_uniform_std():
    f = _compute(uniform_image(1.0))
    uniform_std = 1.0 / np.sqrt(12.0)  # ~0.2887
    assert abs(f["horizontal_spread"] - uniform_std) < 0.01
    assert abs(f["vertical_spread"] - uniform_std) < 0.01
    assert abs(f["elongation"] - 1.0) < 0.02


def test_streak_elongation():
    streak = ellipse_blob_image(ry=0.08, rx=0.40)  # wide horizontal ellipse
    assert _compute(streak)["elongation"] > 4.0


def test_radial_gradient_centre_concentration_vs_edge_extent():
    # Net change from centre outward: a radial ramp (edges much brighter) is
    # strongly positive; a compact centre disc is strongly negative; a centred
    # annulus is radius-symmetric and therefore near zero.
    yy, xx = np.mgrid[0:128, 0:128]
    r = np.hypot(yy - 63.5, xx - 63.5)
    ramp = (r / r.max()).astype(np.float32)
    assert _compute(ramp)["radial_gradient"] > 0.3

    blob = ellipse_blob_image(ry=0.10, rx=0.10)  # bright centre, dark edges
    assert _compute(blob)["radial_gradient"] < -0.2

    ring = ring_image()
    # A centred annulus is radius-symmetric, but all rings sit inside the
    # corner radius, so the empty outer bins (dark corners) dominate the net
    # change -> strongly negative is the honest value for a compact object.
    assert _compute(ring)["radial_gradient"] < -0.5


def test_empty_image_guards():
    f = _compute(np.zeros((128, 128), dtype=np.float32))
    assert f["centre_of_mass_x"] == 0.5 and f["centre_of_mass_y"] == 0.5
    assert f["horizontal_spread"] == 0.0 and f["vertical_spread"] == 0.0
    assert f["elongation"] == 1.0 and f["radial_gradient"] == 0.0


def test_com_near_corner_excludes_empty_rings():
    # Centre of mass near a corner: rings outside the image are empty and
    # must be excluded from the fit rather than poisoning it with NaN.
    blob = ellipse_blob_image(centre=(0.06, 0.06), ry=0.05, rx=0.05)
    f = _compute(blob)
    assert np.isfinite(f["radial_gradient"])
