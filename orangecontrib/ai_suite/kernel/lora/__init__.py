"""MLX LoRA fine-tuning wrappers.

Design rules:

- 4-bit QLoRA only: simple, and the only option that fits a 16 GB classroom
  machine;
- an adapter is serialised with its base-model reference as one artifact, so
  adapter/base mismatches cannot happen;
- models expose the same ``generate(prompt, temperature)`` convention as the
  Keras ``LanguageModel``, so the LM Generator widget is model-agnostic;
- an adapter on/off switch gives the before/after comparison on the identical
  base model;
- MLX is Apple-Silicon-only: every import is guarded, and other machines get a
  clear "not available" message rather than a stack trace.

Training drives the documented ``python -m mlx_lm lora`` command in a
subprocess and streams its metrics. The CLI is a thin, stable wrapper around
the tuner; parsing its output keeps the kernel independent of the tuner's
less-stable internals.
"""

from .model import LoRAAdapter, LoRAModel
from .training import LoraData, LoraOptions, stream_metrics, train_lora

__all__ = [
    "LoRAAdapter",
    "LoRAModel",
    "LoraData",
    "LoraOptions",
    "stream_metrics",
    "train_lora",
]
