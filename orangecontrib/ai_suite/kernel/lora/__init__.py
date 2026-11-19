"""LoRA fine-tuning of an MLX base model, with the adapter stored beside it.

Options, data, the train/validation split, the run and the artifact are here.
:func:`train_lora` returns a run: iterate it for :class:`Params`, :class:`TrainReport`
and :class:`ValReport` events, and read ``run.adapter`` when the iteration ends. The
modules whose names begin ``mlx_`` are the boundary: mlx-lm's files, its child process,
its wire format and its in-process runtime. MLX is Apple-Silicon-only, so those imports
are local and :func:`available` checks the platform before one is tried.
"""

from .artifact import LoRAAdapter
from .data import LoRAData, LoRAOptions, split_for_training
from .mlx_protocol import Params, TrainReport, ValReport
from .mlx_runtime import available as _mlx_available
from .model import LoRAModel
from .run import TrainingRun, TrainingStopped, train_lora


def available() -> bool:
    """Whether this machine can run LoRA at all: MLX is Apple-Silicon-only."""
    return _mlx_available()


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
