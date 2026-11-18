"""Raw pixel values: ordering, shapes, scaling and the requested resolution."""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.images.pixel_values import pixel_values


@pytest.mark.parametrize("size", [1, 3, 10])
def test_values_are_row_major_left_to_right_then_top_to_bottom(size):
    # A grid already at the requested resolution is not resampled, so the values
    # are the picture itself and the ordering can be checked against it directly.
    image = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    values = pixel_values(image, size)
    assert values.shape == (size * size,)
    for index, value in enumerate(values):
        row, column = divmod(index, size)
        assert value == pytest.approx(image[row, column])


def test_pixel_values_shape_and_range():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 255
    out = pixel_values(image, 4)
    assert out.shape == (16,)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() == pytest.approx(1.0)


def test_pixel_values_grow_images_smaller_than_the_requested_resolution():
    image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out = pixel_values(image, 4)  # 2 x 2 -> 4 x 4, so bilinear, not area averaging
    assert out.shape == (16,)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_pixel_values_use_luma():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[..., 0] = 255  # red
    out = pixel_values(image, 1)
    assert out[0] == pytest.approx(0.299, abs=1e-3)


def test_pixel_values_accept_float_zero_to_one():
    image = np.full((4, 4), 0.5, dtype=np.float32)
    out = pixel_values(image, 2)
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_pixel_values_require_a_positive_size():
    with pytest.raises(ValueError):
        pixel_values(np.zeros((4, 4)), 0)


def test_size_is_required():
    with pytest.raises(TypeError):
        pixel_values(np.zeros((4, 4)))
