"""Generate a small, reproducible MNIST subset as PNGs for the Orange demos.

Builds  demo/mnist/{train,val}/<digit>/  trees in the layout Import Images
expects (one subfolder per label). Train: 500/class, val: 100/class = 6,000
images; deterministic seed; digits upscaled x2 to 56x56 (nearest neighbour)
so the feature set has something to chew on (28-wide images starve the
texture features, see demo/README.md).

Run with the development venv (needs the [lm] extra for keras):
    uv run python demo/fetch_mnist.py
MNIST images: widely redistributed; credit per demo/README.md.
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
TARGET = Path(__file__).resolve().parent.parent / "demo" / "mnist"

LABEL_DIRS = {"train": TRAIN_PER_CLASS, "val": VAL_PER_CLASS}


def main() -> None:
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    rng = np.random.default_rng(SEED)
    for split, y in (("train", y_train), ("val", y_val)):
        # stratified pick: seeded shuffle of each class's indices
        chosen: dict[int, list[int]] = defaultdict(list)
        quota = LABEL_DIRS[split]
        for digit in range(10):
            idx = np.flatnonzero(y == digit)
            rng.shuffle(idx)
            chosen[digit] = idx[:quota].tolist()

        split_dir = TARGET / split
        x = x_train if split == "train" else x_val
        written = 0
        for digit, indices in chosen.items():
            out_dir = split_dir / str(digit)
            out_dir.mkdir(parents=True, exist_ok=True)
            for n, i in enumerate(indices):
                img = x[i]
                up = cv2.resize(img, (img.shape[1] * UPSCALE, img.shape[0] * UPSCALE),
                                interpolation=cv2.INTER_NEAREST)
                # PNG from 0..255 uint8
                path = out_dir / f"{digit:07d}_{n:05d}.png"
                if path.exists():
                    continue  # idempotent: keep existing files
                cv2.imwrite(str(path), up)
                written += 1
        print(f"{split}: wrote {written} images ({quota}/class), "
              f"files now on disk: {sum(1 for _ in split_dir.rglob('*.png'))}")

    total = sum(1 for _ in TARGET.rglob("*.png"))
    print(f"total PNGs under {TARGET}: {total}")


if __name__ == "__main__":
    main()