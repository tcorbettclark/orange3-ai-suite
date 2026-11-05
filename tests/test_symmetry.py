"""Group F — symmetry."""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.features import FeatureOptions, symmetry
from synth import band_image, ring_image, uniform_image

OPTS = FeatureOptions()


def _compute(img):
    return symmetry.compute(img, OPTS)


def test_mirror_symmetric_images():
    f = _compute(ring_image())  # annulus centred: symmetric both ways
    assert f["left_right_symmetry"] > 0.99
    assert f["top_bottom_symmetry"] > 0.99

    bar = np.zeros((128, 128), dtype=np.float32)
    bar[:, 58:70] = 1.0  # centred vertical bar
    assert _compute(bar)["left_right_symmetry"] > 0.99


def test_asymmetric():
    horizontal_bands = band_image(((20, 40), (70, 100)))
    f = _compute(horizontal_bands)
    assert f["left_right_symmetry"] > 0.99  # bands span full width
    assert f["top_bottom_symmetry"] < 0.5  # bands are top-heavy


def test_flat_image_is_perfectly_symmetric():
    f = _compute(uniform_image(0.5))
    assert f["left_right_symmetry"] == 1.0
    assert f["top_bottom_symmetry"] == 1.0
    assert f["quadrant_self_similarity"] == 1.0


def test_quadrant_localised_structure_scores_low():
    img = np.zeros((128, 128), dtype=np.float32)
    img[:64, :64] = 1.0  # single bright quadrant, rest dark
    f = _compute(img)
    # three dark-dark quadrant pairs correlate perfectly, three bright-dark
    # pairs hit the constant-mismatch guard (0) -> mean ~0.5, below the
    # ~1.0 of a globally uniform or globally structured image
    assert 0.4 < f["quadrant_self_similarity"] < 0.6


def test_quadrant_correlated_structure_scores_high():
    # The same small blob in each quadrant (same relative offset) makes the
    # quadrants pixel-identical. Note: a centred radially-symmetric disc does
    # NOT score high here — left/right quadrant pairs are mirror images, and
    # correlation is not rotation-invariant. The feature detects
    # translation-similar textures (cloud decks), which is the intended use.
    img = np.zeros((128, 128), dtype=np.float32)
    for cy, cx in ((32, 32), (32, 96), (96, 32), (96, 96)):
        yy, xx = np.mgrid[0:128, 0:128]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 12**2] = 1.0
    f = _compute(img)
    assert f["quadrant_self_similarity"] > 0.95


def test_one_sided_constant_guard():
    # Band image: each half is constant along its own rows but the two halves
    # are pixel-identical (bands span the full width) -> perfect symmetry.
    f = _compute(band_image(((20, 40), (70, 100))))
    assert f["left_right_symmetry"] == pytest.approx(1.0)

    # A constant half against a different constant half is *not* symmetric.
    img = np.zeros((128, 128), dtype=np.float32)
    img[:, :64] = 0.5
    assert _compute(img)["left_right_symmetry"] == 0.0
