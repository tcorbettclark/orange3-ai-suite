"""Dataset preparation: Hub rows -> the folder layout the widgets read.

These functions are deliberately ordinary Python. ``datasets`` has already
downloaded, decoded and cached the source; this is the small domain-specific
step that turns rows into ``train/<label>/<file>`` or ``<genus>/<file>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MNIST_TRAIN_PER_CLASS = 500
MNIST_VAL_PER_CLASS = 100
MNIST_SEED = 0

CCSN_CLASSES = {
    "Ac": "ac",
    "Sc": "sc",
    "Cu": "cu",
    "Cb": "cb",
    "Ci": "ci",
}

MNIST_README = """# MNIST digits

A deterministic subset of [MNIST](https://huggingface.co/datasets/ylecun/mnist)
(MIT): 500 images per digit for training and 100 per digit for validation,
kept at the source 28x28 resolution. The folder-per-label layout is what
Orange's Import Images expects.
"""

CCSN_CREDITS = """# CCSN cloud images

A five-genus subset of the
[Cirrus Cumulus Stratus Nimbus (CCSN) database](https://doi.org/10.7910/DVN/CADDPD):
Altocumulus (Ac), Stratocumulus (Sc), Cumulus (Cu), Cumulonimbus (Cb) and
Cirrus (Ci). Images are 256x256 JPEGs.

The CCSN database is released under CC0 1.0. The Hugging Face mirror used here
is [aduuuuuu/CCSN](https://huggingface.co/datasets/aduuuuuu/CCSN) (MIT). Please
cite:

> Zhang, J. L., Liu, P., Zhang, F., & Song, Q. Q. (2018). CloudNet: Ground-based
> cloud classification with deep convolutional neural network. Geophysical
> Research Letters, 45, 8665-8672. https://doi.org/10.1029/2018GL077787
"""


def _write_split(
    split: Any,
    dest: Path,
    quota: int,
    seed: int,
    *,
    digits: int = 10,
) -> int:
    rng = np.random.default_rng(seed)
    labels = [int(label) for label in split["label"]]
    written = 0
    for digit in range(digits):
        indices = [i for i, label in enumerate(labels) if label == digit]
        rng.shuffle(indices)
        chosen = indices[:quota]
        out_dir = dest / str(digit)
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, index in enumerate(chosen):
            image = split[index]["image"]
            image.save(out_dir / f"{digit:07d}_{n:05d}.png")
            written += 1
    return written


def prepare_mnist(data: Any, dest: Path) -> None:
    """Write the 500/100-per-digit, 28x28 folder tree from ``ylecun/mnist``."""
    dest.mkdir(parents=True, exist_ok=True)
    _write_split(data["train"], dest / "train", MNIST_TRAIN_PER_CLASS, MNIST_SEED)
    _write_split(data["test"], dest / "val", MNIST_VAL_PER_CLASS, MNIST_SEED)
    (dest / "README.md").write_text(MNIST_README)


def prepare_ccsn(data: Any, dest: Path) -> None:
    """Write the five-genus folder tree and credits from ``aduuuuuu/CCSN``."""
    split = data["train"]
    names = split.features["label"].names
    dest.mkdir(parents=True, exist_ok=True)
    for genus in CCSN_CLASSES.values():
        (dest / genus).mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(split):
        label = names[int(example["label"])]
        genus = CCSN_CLASSES.get(label)
        if genus is None:
            continue
        image = example["image"]
        image.save(dest / genus / f"{index:05d}.jpg", format="JPEG", quality=88)
    (dest / "CREDITS.md").write_text(CCSN_CREDITS)
