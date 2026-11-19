"""LoRA fine-tuning of an MLX base model, with the adapter stored beside it.

- QLoRA: a quantised base model plus a low-rank adapter (the course uses
  the 4-bit BritLLM model);
- an adapter is serialised with its base-model reference as one artifact, so
  the two travel together;
- ``LoRAModel`` exposes the kernel's ``generate`` and ``evaluate`` conventions,
  shared with the Keras ``LanguageModel``;
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
from .model import LoRAAdapter, LoRAModel
from .training import (
    LoRAData,
    LoRAOptions,
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
    "TrainingStopped",
    "available",
    "split_for_training",
    "train_lora",
]
