"""LoRA fine-tuning of an MLX base model, with the adapter stored beside it.

Options, data, the train/validation split, the run and the artifact live here.
:func:`train_lora` returns a run: iterate it for :class:`Params`,
:class:`TrainReport` and :class:`ValReport` events, and read ``run.adapter`` when
the iteration ends.
everything that has to know mlx-lm is behind :mod:`~.lora.mlx`. MLX is
Apple-Silicon-only, so its imports are local and :func:`available` answers before one
is attempted.
"""

from . import mlx
from .mlx import Params, TrainReport, ValReport
from .model import LoRAAdapter, LoRAModel
from .training import (
    LoRAData,
    LoRAOptions,
    TrainingRun,
    TrainingStopped,
    split_for_training,
    train_lora,
)


def available() -> bool:
    """Whether this machine can run LoRA at all: MLX is Apple-Silicon-only."""
    return mlx.available()


__all__ = [
    "LoRAAdapter",
    "LoRAData",
    "LoRAModel",
    "LoRAOptions",
    "Params",
    "TrainReport",
    "TrainingRun",
    "TrainingStopped",
    "ValReport",
    "available",
    "split_for_training",
    "train_lora",
]
