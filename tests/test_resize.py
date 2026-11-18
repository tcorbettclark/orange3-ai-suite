"""Resampling: one policy, area when shrinking and bilinear when growing."""

import cv2
import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.images.resize import resize_square


def test_resize_stretches_the_whole_frame_to_a_square():
    image = np.linspace(0.0, 1.0, 40 * 20, dtype=np.float32).reshape(20, 40)
    assert resize_square(image, 8).shape == (8, 8)


@pytest.mark.parametrize("source, size", [(40, 8), (8, 40)])
def test_resize_uses_area_when_shrinking_and_bilinear_when_growing(source, size):
    image = np.linspace(0.0, 1.0, source * source, dtype=np.float32).reshape(source, source)
    expected = cv2.INTER_AREA if size < source else cv2.INTER_LINEAR  # the policy this module owns
    assert np.allclose(
        resize_square(image, size), cv2.resize(image, (size, size), interpolation=expected)
    )


def test_resize_keeps_already_square_input_unresampled():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    out = resize_square(image, 2)
    assert out.dtype == np.float32
    assert np.array_equal(out, image)


def test_resize_carries_channels():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    assert resize_square(rgb, 4).shape == (4, 4, 3)


@pytest.mark.parametrize("size", [0, -3, True, 2.5, "8"])
def test_resize_rejects_an_invalid_size(size):
    with pytest.raises(ValueError):
        resize_square(np.zeros((4, 4), np.float32), size)


def test_resize_rejects_an_empty_image():
    with pytest.raises(ValueError):
        resize_square(np.zeros((0, 4), np.float32), 2)
