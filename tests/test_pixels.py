"""The plain pixel baseline: decode, resize, flatten as one call."""

import numpy as np
import pytest

from edu_ai_suite.images.pixels import pixel_row


def _rgb(height, width, value=0.5):
    return np.full((height, width, 3), value, dtype=np.float32)


def test_a_row_is_square_and_named_by_coordinate():
    values, names = pixel_row(_rgb(10, 40), 4)
    assert values.shape == (16,)
    assert values.dtype == np.float32
    assert names == tuple(f"pixel_r{r}_c{c}" for r in range(4) for c in range(4))


def test_the_grid_size_is_the_only_thing_that_sets_the_count():
    for size in (1, 4, 10):
        values, names = pixel_row(_rgb(7, 13), size)
        assert values.size == size * size == len(names)


def test_values_are_luminance_in_the_unit_range():
    # A pure red pixel is 0.299 of the luma, whatever the exposure: brightness
    # is taken before the value range is normalised, so the row is comparable.
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[..., 0] = 1.0
    values, _ = pixel_row(image, 2)
    assert np.allclose(values, 0.299, atol=1e-3)

    dark = _rgb(4, 4, 0.0)
    bright = _rgb(4, 4, 1.0)
    assert np.allclose(pixel_row(dark, 2)[0], 0.0)
    assert np.allclose(pixel_row(bright, 2)[0], 1.0)


def test_grayscale_input_needs_no_channel_handling():
    values, names = pixel_row(np.full((4, 4), 0.5, dtype=np.float32), 2)
    assert values.shape == (4,)
    assert len(names) == 4


def test_the_frame_is_stretched_not_cropped():
    # A one-pixel stripe on the left edge stays at the left edge: nothing is
    # centre-cropped on the way to a square.
    image = np.zeros((8, 40), dtype=np.float32)
    image[:, 0] = 1.0
    values, _ = pixel_row(image, 10)  # 40 source columns -> 10 cells of 4 pixels each
    grid = values.reshape(10, 10)
    assert grid[:, 0].min() == pytest.approx(0.25)  # the stripe survives, on column 0
    assert np.allclose(grid[:, 1:], 0.0)


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_a_non_positive_or_non_integer_size_is_refused(size):
    with pytest.raises(ValueError, match="size must be a positive integer"):
        pixel_row(_rgb(4, 4), size)


def test_an_empty_image_is_refused():
    with pytest.raises(ValueError, match="empty image"):
        pixel_row(np.zeros((0, 0), dtype=np.float32), 4)
