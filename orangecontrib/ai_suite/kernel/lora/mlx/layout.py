"""mlx-lm's file conventions, in one place.

The library's API is one thing to track; its *files* are another, and they were
spread over three modules of the lora package before this one existed: the jsonl
shapes mlx-lm reads, the name it gives the final adapter, and the numbered
checkpoints it leaves behind when a run is stopped.

Nothing here imports mlx or mlx_lm: these are names and formats, which is why
they can be tested without a model, a machine with Apple silicon, or a run.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The adapter files mlx-lm writes, and the metadata our artifacts carry.
ADAPTER_WEIGHTS = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

#: The two dataset shapes mlx-lm reads: raw text, and prompt/completion pairs.
_FORMATS = {"text": ("text",), "pairs": ("prompt", "completion")}


def write_dataset(directory: str | Path, format: str, train: list, valid: list) -> Path:
    """Write ``{train,valid}.jsonl`` in the shape mlx-lm reads, and return the directory.

    ``valid.jsonl`` is omitted when there is no validation split: mlx-lm treats a
    missing file as no validation but fails on an empty one. A stale file from an
    earlier run is removed, so it cannot be picked up by mistake.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = _FORMATS[format]
    splits = [("train", train), ("valid", valid)] if valid else [("train", train)]
    if not valid:
        stale = directory / "valid.jsonl"
        if stale.exists():
            stale.unlink()
    for split, rows in splits:
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                record = (
                    {fields[0]: row} if len(fields) == 1 else dict(zip(fields, row, strict=True))
                )
                handle.write(json.dumps(record) + "\n")
    return directory


def adapter_weights(adapter_dir: str | Path) -> Path | None:
    """The final adapter if mlx-lm wrote it, else its newest numbered checkpoint.

    A stopped run leaves ``{iteration:07d}_adapters.safetensors`` files and no
    final adapter, so the newest is what the run actually reached.
    """
    adapter_dir = Path(adapter_dir)
    canonical = adapter_dir / ADAPTER_WEIGHTS
    if canonical.exists():
        return canonical
    checkpoints: list[tuple[int, Path]] = []
    for path in adapter_dir.glob(f"*_{ADAPTER_WEIGHTS}"):
        try:
            checkpoints.append((int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def checkpoint_iteration(weights: Path) -> int | None:
    """The iteration a numbered checkpoint belongs to, or None for the final adapter."""
    if weights.name == ADAPTER_WEIGHTS:
        return None
    try:
        return int(weights.name.split("_", 1)[0])
    except ValueError:
        return None
