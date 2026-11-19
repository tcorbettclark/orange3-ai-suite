"""LoRA fine-tuning of an MLX base model, with the adapter stored beside it.

- QLoRA: a quantised base model plus a low-rank adapter (the course uses
  the 4-bit BritLLM model);
- an adapter is serialised with its base-model reference as one artifact, so
  the two travel together;
- ``LoRAModel`` exposes the kernel's ``stream``/``generate``/``evaluate``
  conventions, shared with the Keras ``LanguageModel``;
- a run is an iterator: ``for event in train_lora(...)`` reports progress as
  :class:`Params`, :class:`TrainReport` and :class:`ValReport`, and the artifact
  is ready as ``run.adapter`` when the iteration ends;
- ``adapter_enabled`` selects the raw base model or the adapted one, on the
  same weights, for the before/after comparison;
- MLX is Apple-Silicon-only: imports are local, and other machines get the
  ``[lora]`` install hint.

This package is the domain: options, data, the split, the run's scratch
directory, the artifact, and the words a teaching run is explained in. Everything
that has to know mlx-lm - its API, its file conventions, and the fact that
training happens in another process - is behind ``lora.mlx``.
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
