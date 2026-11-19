"""The files mlx-lm reads and writes.

The jsonl shapes it reads, the name it gives the final adapter, and the numbered
checkpoints a run leaves behind when it is stopped.

Nothing here imports mlx or mlx_lm: these are names and formats.
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
    """The file to load as the adapter: the final one, else the newest checkpoint.

    A stopped run leaves ``{iteration:07d}_adapters.safetensors`` files and no final
    adapter, so the newest is what the run actually reached.
    """
    adapter_dir = Path(adapter_dir)
    canonical = adapter_dir / ADAPTER_WEIGHTS
    if canonical.exists():
        return canonical
    found = checkpoints(adapter_dir)
    return found[-1][1] if found else None


def checkpoints(adapter_dir: str | Path) -> list[tuple[int, Path]]:
    """The numbered checkpoints a run leaves behind, oldest first: ``(iteration, path)``.

    mlx-lm writes the unnumbered adapter on every save as well as the numbered file,
    so a run that is still going has both; only these say how far it got.
    """
    found: list[tuple[int, Path]] = []
    for path in Path(adapter_dir).glob(f"*_{ADAPTER_WEIGHTS}"):
        try:
            found.append((int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    return sorted(found)
