"""Group G — colour (the trap): shortcut features, default off."""

import numpy as np

from nn_kernel.features import FeatureOptions, colour, extract_features

OPTS = FeatureOptions()


def _compute(img):
    return colour.compute(img, OPTS)


def sky_photo(size=64):
    """Blue sky with white clouds (the trap case)."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 2] = 0.8  # blue everywhere
    img[..., :2] = 0.5
    yy, xx = np.mgrid[0:size, 0:size]
    img[(yy - 20) ** 2 + (xx - 30) ** 2 < 60] = [0.95, 0.95, 0.98]  # white cloud
    return img


def field_photo(size=64):
    """Ground: grass green everywhere (no sky)."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0], img[..., 1], img[..., 2] = (0.25, 0.55, 0.20)
    return img


def test_bluish_fraction_separates_sky_from_ground():
    sky = _compute(sky_photo())
    field = _compute(field_photo())
    assert sky["bluish_fraction"] > 0.5
    assert field["bluish_fraction"] == 0.0
    assert field["greenery_fraction"] > 0.95
    assert sky["greenery_fraction"] < 0.05


def test_traps_are_documented_as_shortcuts():
    definition = colour.DEFINITIONS["bluish_fraction"].lower()
    assert "shortcut" in definition  # the lesson is part of the data


def test_grayscale_input_gives_zeros():
    gray = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    assert all(v == 0.0 for v in _compute(gray).values())


def test_rgba_input_alpha_dropped():
    rgba = np.concatenate([sky_photo(), np.ones((64, 64, 1), np.float32)], axis=-1)
    f_rgb = _compute(sky_photo())
    f_rgba = _compute(rgba)
    assert f_rgba["bluish_fraction"] == f_rgb["bluish_fraction"]


def test_extraction_includes_colour_only_when_asked():
    img = sky_photo()
    default = extract_features(img)
    assert "bluish_fraction" not in default
    with_colour = extract_features(img, groups="ABCDEFG")
    assert with_colour["bluish_fraction"] > 0.5
    assert with_colour.group_of("bluish_fraction") == "G"


def test_colour_group_off_by_default():
    from nn_kernel.features import DEFAULT_GROUPS

    assert "G" not in DEFAULT_GROUPS
