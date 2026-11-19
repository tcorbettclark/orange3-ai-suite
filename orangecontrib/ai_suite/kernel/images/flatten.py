"""A 2-D grid as one row, with the coordinate of every cell.

Position ``index`` is row ``index // columns`` and column ``index % columns``,
left to right then top to bottom.
"""

from __future__ import annotations

import numpy as np


def flatten(grid: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Flatten a 2-D grid row-major and name each cell by its coordinate.

    Each name is ``pixel_r{row}_c{column}``, the form a table uses for its
    columns.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("empty grid")
    rows, columns = grid.shape
    names = tuple(f"pixel_r{row}_c{column}" for row in range(rows) for column in range(columns))
    return np.asarray(grid, dtype=np.float32).reshape(-1), names
