"""Flattening: one row per grid, and a name for every cell."""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.images.flatten import flatten


@pytest.mark.parametrize("size", [1, 3, 10])
def test_flatten_is_row_major_and_names_are_coordinates(size):
    grid = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    values, names = flatten(grid)
    assert values.shape == (size * size,)
    assert len(names) == size * size
    for index, (value, name) in enumerate(zip(values, names, strict=True)):
        row, column = divmod(index, size)
        assert value == pytest.approx(grid[row, column])
        assert name == f"pixel_r{row}_c{column}"


def test_flatten_names_a_non_square_grid_correctly():
    values, names = flatten(np.zeros((2, 3), np.float32))
    assert len(values) == len(names) == 6
    assert names[2:5] == ("pixel_r0_c2", "pixel_r1_c0", "pixel_r1_c1")


def test_flatten_returns_float32_values():
    values, _ = flatten(np.zeros((2, 2), np.uint8))
    assert values.dtype == np.float32


@pytest.mark.parametrize("bad", [np.zeros(3, np.float32), np.zeros((2, 2, 3), np.float32)])
def test_flatten_rejects_non_2d_input(bad):
    with pytest.raises(ValueError, match="2-D"):
        flatten(bad)
