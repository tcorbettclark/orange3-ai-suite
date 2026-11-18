"""MNIST digits as a prepared Hub dataset: 28x28, 500/100 per digit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hf import Asset, Progress, ensure

MNIST_TRAIN_PER_CLASS = 500
MNIST_VAL_PER_CLASS = 100
MNIST_SEED = 0

_README = """# MNIST digits

A deterministic subset of [MNIST](https://huggingface.co/datasets/ylecun/mnist)
(MIT): 500 images per digit for training and 100 per digit for validation,
kept at the source 28x28 resolution. The folder-per-label layout is what
Orange's Import Images expects.
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


def _prepare_mnist(data: Any, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _write_split(data["train"], dest / "train", MNIST_TRAIN_PER_CLASS, MNIST_SEED)
    _write_split(data["test"], dest / "val", MNIST_VAL_PER_CLASS, MNIST_SEED)
    (dest / "README.md").write_text(_README)


MNIST_DIGITS = Asset(
    name="mnist-digits",
    repo_id="ylecun/mnist",
    repo_type="dataset",
    revision="77f3279092a1c1579b2250db8eafed0ad422088c",
    size_bytes=18_157_506,
    required=("train", "val", "README.md"),
    prepare=_prepare_mnist,
)


def get_mnist(
    cache_dir: str | Path | None = None,
    on_progress: Progress | None = None,
) -> Path:
    """Download, prepare and return the 28x28 MNIST subset."""
    return ensure(MNIST_DIGITS, cache_dir, on_progress)
