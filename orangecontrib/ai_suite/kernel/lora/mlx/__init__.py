"""The MLX boundary: the one place the LoRA kernel reaches outside itself.

:mod:`runtime` loads, generates and scores through mlx-lm, :mod:`layout` holds that
library's file names and jsonl shapes, and :mod:`train` runs training as a child
process behind an iterator, on the wire format of :mod:`protocol`. ``lora`` may import
this package and nothing above it may, nor does this package import ``lora`` back.
"""

from __future__ import annotations

from .layout import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    adapter_weights,
    checkpoint_iteration,
    write_dataset,
)
from .protocol import Event, Params, TrainReport, ValReport
from .runtime import available, evaluate_text, load_model, require, stream_text
from .train import Outcome, Run, TrainingFailed, TrainingSpec

__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "Event",
    "Outcome",
    "Params",
    "Run",
    "TrainReport",
    "TrainingFailed",
    "TrainingSpec",
    "ValReport",
    "adapter_weights",
    "available",
    "checkpoint_iteration",
    "evaluate_text",
    "load_model",
    "require",
    "stream_text",
    "write_dataset",
]
