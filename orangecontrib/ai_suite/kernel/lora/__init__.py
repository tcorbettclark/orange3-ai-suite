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

Training runs the documented ``python -m mlx_lm lora`` command in a subprocess
and parses its output, so the kernel depends on the CLI rather than on the
tuner's internals.
"""

from .model import LoRAAdapter, LoRAModel
from .training import LoRAData, LoRAOptions, TrainingStopped, parse_metric, train_lora

__all__ = [
    "LoRAAdapter",
    "LoRAModel",
    "LoRAData",
    "LoRAOptions",
    "TrainingStopped",
    "parse_metric",
    "train_lora",
]
