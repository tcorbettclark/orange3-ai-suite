"""Generate a small, reproducible MNIST subset as PNGs for the Orange demos.

Builds ``demo/mnist/{train,val}/<digit>/`` trees in the layout Import Images
expects (one subfolder per label). Train: 500/class, validation: 100/class,
6,000 images in total; deterministic seed. Digits are upscaled 2x to 56x56
(nearest neighbour) so the texture features have something to measure -
28-wide images starve the 5x5 and 16x16 texture windows.

    uv run python demo/fetch_mnist.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from keras.datasets import mnist

SEED = 0
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
UPSCALE = 2  # 28 -> 56
TARGET = Path(__file__).resolve().parent / "mnist"


def main() -> None:
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    rng = np.random.default_rng(SEED)

    for split, images, labels, quota in (
        ("train", x_train, y_train, TRAIN_PER_CLASS),
        ("val", x_val, y_val, VAL_PER_CLASS),
    ):
        # stratified pick: seeded shuffle of each class's indices
        chosen: dict[int, list[int]] = defaultdict(list)
        for digit in range(10):
            idx = np.flatnonzero(labels == digit)
            rng.shuffle(idx)
            chosen[digit] = idx[:quota].tolist()

        out_dir = TARGET / split
        written = 0
        for digit, indices in chosen.items():
            digit_dir = out_dir / str(digit)
            digit_dir.mkdir(parents=True, exist_ok=True)
            for n, i in enumerate(indices):
                image = images[i]
                upscaled = cv2.resize(
                    image,
                    (image.shape[1] * UPSCALE, image.shape[0] * UPSCALE),
                    interpolation=cv2.INTER_NEAREST,
                )
                path = digit_dir / f"{digit:07d}_{n:05d}.png"
                if path.exists():
                    continue  # idempotent: keep existing files
                cv2.imwrite(str(path), upscaled)
                written += 1
        print(f"{split}: wrote {written} images ({quota}/class)")

    total = sum(1 for _ in TARGET.rglob("*.png"))
    print(f"total PNGs under {TARGET}: {total}")


if __name__ == "__main__":
    main()
