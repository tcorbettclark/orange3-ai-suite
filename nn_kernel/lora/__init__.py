"""nn_kernel.lora: MLX LoRA fine-tuning wrappers.

Design commitments (spec/tiny-lm-suite-design.md §4.3, architecture doc):
- 4-bit QLoRA only — simplicity + the 16 GB classroom constraint;
- adapter serialized with its base-model reference: ONE artifact, so the
  adapter/base mismatch bug is unrepresentable (same as tokenizer-with-model);
- the same ``generate(prompt, temperature)`` convention as the Keras
  LanguageModel — Stages 2-3 share the LM Generator widget unchanged;
- adapter on/off toggle = the clean before/after comparison of the same
  model;
- MLX is Apple-Silicon-only: every import is guarded; other machines get a
  clear "not available" message, never a stack trace.

Implementation note (documented deviation): training drives the *documented*
`python -m mlx_lm lora` command (subprocess, streamed metrics) rather than
the tuner's in-process trainer callbacks — the tuner internals are the least
stable part of the stack (the spec's own pinning warning), while the CLI is
a thin stable wrapper; streaming + stop handling are trivial at this layer.

Background provenance: hyperparameter defaults (iters 300, lr 1e-4,
16 layers, batch 4, seq 512) come from reference runs elsewhere; the repo
records its own dry-run measurements in demo/lora_dry_run.py output.
"""

from .model import LoRAAdapter, LoRAModel
from .training import LoraData, LoraOptions, stream_metrics, train_lora

__all__ = [
    "LoRAAdapter",
    "LoRAModel",
    "LoraOptions",
    "LoraData",
    "train_lora",
    "stream_metrics",
]