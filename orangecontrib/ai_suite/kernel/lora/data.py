"""The rows a run trains on and the settings it trains with, before the split.

Two formats: rows of raw text for continued pre-training, whose next-token loss is
self-supervised, and prompt/completion pairs for supervised fine-tuning, where the
prompt tokens are masked out of the loss. :func:`split_for_training` divides the rows,
by position in the corpus rather than by content.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .._holdout import held_out_positions

#: "text" is continued pre-training and "pairs" is supervised fine-tuning.
Format = Literal["text", "pairs"]
Rows = list[str] | list[tuple[str, str]]


@dataclass(frozen=True)
class LoRAOptions:
    """Hyperparameters for one LoRA run on an MLX base model."""

    iters: int = 300
    learning_rate: float = 1e-4
    num_layers: int = 16
    batch_size: int = 4
    max_seq_length: int = 512
    validation_fraction: float = 0.1
    steps_per_report: int = 20
    steps_per_eval: int = 50
    save_every: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "iters",
            "num_layers",
            "batch_size",
            "steps_per_report",
            "steps_per_eval",
            "save_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be >= 2")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True)
class LoRAData:
    """The rows of one of the two formats, before the train/validation split."""

    format: Format  # "text" (continued pre-training) | "pairs" (SFT)
    rows: Rows

    def __post_init__(self) -> None:
        if self.format not in ("text", "pairs"):
            raise ValueError(f"unknown format {self.format!r}: expected 'text' or 'pairs'")
        if self.format == "pairs":
            if not all(
                isinstance(row, tuple) and len(row) == 2 and all(isinstance(v, str) for v in row)
                for row in self.rows
            ):
                raise TypeError("pair rows must be (prompt, completion) string tuples")
        elif not all(isinstance(row, str) for row in self.rows):
            raise TypeError("text rows must be strings")

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> LoRAData:
        rows = [t for t in texts if t and t.strip()]
        if len(rows) < 2:
            raise ValueError("need at least 2 texts (train + validation)")
        return cls("text", rows)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> LoRAData:
        rows = [(str(p), str(c)) for p, c in pairs if str(p).strip() and str(c).strip()]
        if len(rows) < 4:
            raise ValueError("need at least 4 prompt-completion pairs (SFT)")
        return cls("pairs", rows)

    def split(self, valid_fraction: float, seed: int) -> tuple[Rows, Rows]:
        """Train and validation rows, chosen by position in the corpus.

        :func:`.._holdout.held_out_positions` picks the position; this method applies
        it to rows. ``0`` means no validation at all, and a corpus of two or more rows
        keeps at least one on each side. A row's side depends on where it sits, not on its
        content, so identical rows may land on different sides and reordering the rows
        can move one across.
        """
        if not 0.0 <= valid_fraction < 1.0:
            raise ValueError("valid_fraction must be in [0, 1)")
        if valid_fraction == 0.0:
            return list(self.rows), []
        held = set(held_out_positions(len(self.rows), valid_fraction, seed))
        train = [row for index, row in enumerate(self.rows) if index not in held]
        # sorted: the rows come back in corpus order, not in set order
        return train, [self.rows[index] for index in sorted(held)]


def split_for_training(data: LoRAData, options: LoRAOptions) -> tuple[Rows, Rows]:
    """The train and validation rows, refusing a validation split mlx-lm cannot use.

    A non-empty validation split holds at least ``batch_size`` rows: mlx-lm fails on
    anything less. An empty one is allowed, and mlx-lm then trains without validation.
    """
    train, valid = data.split(options.validation_fraction, options.seed)
    if valid and len(valid) < options.batch_size:
        rows = "row" if len(valid) == 1 else "rows"
        raise ValueError(
            f"valid split has {len(valid)} {rows} but batch_size is "
            f"{options.batch_size} - mlx-lm requires the valid split to hold at "
            "least one full batch; add data, reduce batch_size or raise "
            "validation_fraction"
        )
    return train, valid
