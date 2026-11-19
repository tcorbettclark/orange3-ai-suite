"""The fine-tuning artifact: the adapter files, and the base model they fit.

A :class:`LoRAAdapter` is a directory holding the adapter weights, its config and a
``lora_meta.json`` with the base-model identity and the run that produced it. Reading
and writing it is plain Python: no mlx import, so an artifact can be inspected and
reshaped on a machine that cannot train.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import mlx_layout

_META_FILE = "lora_meta.json"


@dataclass
class LoRAAdapter:
    """The fine-tuning artifact: the adapter files, with the base model they fit."""

    base_model: str  # path or HF repo id of the base model
    adapter_path: str | None  # None = the raw base model itself (no adapter)
    training: dict[str, Any]  # options, mlx-lm adapter config, iterations reached
    name: str = ""

    def save(self, dest: str | Path) -> Path:
        """Write the artifact directory and return its path.

        An ``adapter_path`` with no weights file in it raises before anything is
        written.
        """
        src = Path(self.adapter_path) if self.adapter_path is not None else None
        if src is not None and not (src / mlx_layout.ADAPTER_WEIGHTS).is_file():
            raise FileNotFoundError(f"no {mlx_layout.ADAPTER_WEIGHTS} in {src}")
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        if src is not None:
            for fname in (mlx_layout.ADAPTER_WEIGHTS, mlx_layout.ADAPTER_CONFIG):
                if (src / fname).exists():
                    shutil.copy2(src / fname, out / fname)
        (out / _META_FILE).write_text(
            json.dumps(
                {
                    "base_model": self.base_model,
                    "training": self.training,
                    "name": self.name,
                },
                indent=1,
            )
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> LoRAAdapter:
        """Read an adapter directory, or a raw model directory as the base model.

        An artifact carries ``lora_meta.json``; a directory without it is a plain
        MLX model directory, loaded with the adapter off.
        """
        src = Path(path)
        if (src / _META_FILE).exists():
            meta = json.loads((src / _META_FILE).read_text())
            has_weights = (src / mlx_layout.ADAPTER_WEIGHTS).exists()
            return cls(
                base_model=meta["base_model"],
                adapter_path=str(src) if has_weights else None,
                training=meta["training"],
                name=meta.get("name", ""),
            )
        return cls(base_model=str(src), adapter_path=None, training={})

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"LoRAAdapter(base={self.base_model}, adapter={'yes' if self.adapter_path else 'off'})"
        )
