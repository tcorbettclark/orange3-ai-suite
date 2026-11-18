"""CCSN cloud genera as a prepared Hub dataset: five genera, 256x256."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hf import Asset, Progress, ensure

CCSN_CLASSES = {
    "Ac": "ac",
    "Sc": "sc",
    "Cu": "cu",
    "Cb": "cb",
    "Ci": "ci",
}

_CREDITS = """# CCSN cloud images

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


def _prepare_clouds(data: Any, dest: Path) -> None:
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
    (dest / "CREDITS.md").write_text(_CREDITS)


CCSN_CLOUDS = Asset(
    name="ccsn-clouds",
    repo_id="aduuuuuu/CCSN",
    repo_type="dataset",
    revision="cc195cf92fbb4b668b7db0611db8c1015c7f5cd4",
    size_bytes=103_773_179,
    required=("ac", "sc", "cu", "cb", "ci", "CREDITS.md"),
    prepare=_prepare_clouds,
)


def get_clouds(
    cache_dir: str | Path | None = None,
    on_progress: Progress | None = None,
) -> Path:
    """Download, prepare and return the five-genus CCSN cloud subset."""
    return ensure(CCSN_CLOUDS, cache_dir, on_progress)
