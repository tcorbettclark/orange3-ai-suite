"""One photograph as raw pixel values.

Run it from the repository root:

    uv run python examples/image_pixel_values.py [photo] [grid size]

``kernel/images/pixel_values.py`` is the plain baseline: one image array in, one
flat vector of luma values out. It does not read files, name columns, build tables
or handle batches - the Image Pixels widget supplies those. This script shows the
three things the module does guarantee:

* values are luma in [0, 1], in row-major order;
* there are exactly ``size * size`` of them, and the grid size is passed in;
* the grid is the whole frame resized straight to a square, with no centre crop
  and no contrast stretch - a dark photograph stays dark.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from orangecontrib.ai_suite.kernel.images.pixel_values import pixel_values

HERE = Path(__file__).parent
RAMP = " .:-=+*#%@"  # ten steps from dark to bright


def column_names(size: int) -> list[str]:
    """What the Image Pixels widget calls the cells: a coordinate, not a concept."""
    return [f"pixel_r{row}_c{column}" for row in range(size) for column in range(size)]


def load(path: Path) -> np.ndarray:
    """Read a photo as RGB. The kernel never guesses channel order."""
    if not path.is_file() and (HERE / path).is_file():
        path = HERE / path  # a bare name means one of the photos in this directory
    if not path.is_file():
        raise SystemExit(f"cannot read {path}")
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_grid(values: np.ndarray, size: int) -> None:
    grid = values.reshape(size, size)
    for row in grid:
        print("    " + " ".join(f"{value:.2f}" for value in row))
    print(f"\n    the same grid as characters, {RAMP!r} from dark to bright:")
    for row in grid:
        print(
            "    " + "".join(RAMP[min(int(value * len(RAMP)), len(RAMP) - 1)] * 2 for value in row)
        )


def main(argv: list[str]) -> None:
    path = Path(argv[0]) if argv else HERE / "cirrus.jpg"
    size = int(argv[1]) if len(argv) > 1 else 10
    image = load(path)

    height, width = image.shape[:2]
    print(
        f"image        {path.name}  {width} x {height} pixels, RGB, aspect {width / height:.2f} : 1"
    )
    print(f"grid         {size} x {size}, aspect 1.00 : 1 (the frame is stretched, not cropped)")
    resampling = "INTER_AREA (shrinking)" if size < min(height, width) else "INTER_LINEAR"
    print(f"resampling   {resampling}")

    values = pixel_values(image, size)
    names = column_names(size)
    print(f"\n  pixel_values(image, {size})")
    print(f"    shape {values.shape}, dtype {values.dtype}")
    print(f"    values {values.min():.3f} .. {values.max():.3f}, mean {values.mean():.3f}")
    print(f"    names  {len(names)}, first {names[0]!r}, last {names[-1]!r}")

    print("\n  the grid, dark = 0, bright = 1:")
    show_grid(values, size)

    print("\n  the vector is that grid, read row by row:")
    print(f"    values[:{size}]         {np.array2string(values[:size], precision=2)}")
    print(
        f"    values[{size}:{2 * size}]       {np.array2string(values[size : 2 * size], precision=2)}"
    )
    print(f"    names[:{size}]          {', '.join(names[:size])}")
    print(
        f"\n    index = row * {size} + column, so values[{2 * size + 1}] "
        f"is row 2, column 1: {values[2 * size + 1]:.2f} ({names[2 * size + 1]})"
    )

    print("\n  the same photograph at three resolutions:")
    for other in (4, 10, 20):
        coarse = pixel_values(image, other)
        print(
            f"    {other:2d} x {other:<2d} -> {len(column_names(other)):3d} columns, "
            f"mean {coarse.mean():.3f}"
        )
    print("\n    at 4 x 4 the same frame is that coarse:")
    show_grid(pixel_values(image, 4), 4)
    print("    fewer cells keep the overall brightness and lose the detail; the count")
    print("    grows as the square of the side, which is why the widget defaults to 10.")

    print("\n  not stretched: the range above comes from the input's own scale")
    print("  (255 -> 1.0), not from this photograph's darkest and brightest cell.")


if __name__ == "__main__":
    main(sys.argv[1:])
