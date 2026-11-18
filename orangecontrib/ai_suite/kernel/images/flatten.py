"""A 2-D grid as one row, with the coordinate of every cell.

The values and their names come out of the same call, so they cannot disagree:
position ``index`` is row ``index // columns`` and column ``index % columns``,
left to right then top to bottom.
"""

from __future__ import annotations

import numpy as np


def flatten(grid: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Flatten a 2-D grid row-major and name each cell by its coordinate.

    The names are what a table calls its columns - ``pixel_r4_c7`` is row 4,
    column 7 - not a description of anything. They are derived from the grid's
    own shape, so a non-square grid is named correctly too.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("empty grid")
    rows, columns = grid.shape
    names = tuple(f"pixel_r{row}_c{column}" for row in range(rows) for column in range(columns))
    return np.asarray(grid, dtype=np.float32).reshape(-1), names
