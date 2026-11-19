"""The MLX boundary: the one place the LoRA kernel reaches outside itself.

Above this package sits domain code - options, data, the split, the run's scratch
directory, the artifact - and it imports nothing from ``mlx`` or ``mlx_lm``. This
package holds everything that does, and hides three things that are not ours to
invent:

- **the library's API** (:mod:`runtime`, :mod:`child`): loading, generating,
  scoring, and the tuner's entry points and callbacks;
- **its file conventions** (:mod:`layout`): the jsonl shapes it reads, the names
  it gives the adapter and its numbered checkpoints;
- **the fact that training is another process** (:mod:`train`): a :class:`Run` is
  iterated for progress events and stopped with ``stop()``, like any other
  in-process iterator. The child, the wire format (:mod:`protocol`), the stop
  signal and the wait-and-reap behind it are invisible from outside.

``lora`` may import this package; nothing above ``lora`` may, and nothing in here
may import ``lora`` back. ``tests/test_layout.py`` enforces both.
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
from .runtime import (
    EVAL_WINDOW,
    available,
    evaluate_text,
    generate_text,
    load_model,
    require,
    segments,
    stream_text,
)
from .train import Outcome, Run, TrainingFailed, TrainingSpec, command

__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "EVAL_WINDOW",
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
    "command",
    "evaluate_text",
    "generate_text",
    "load_model",
    "require",
    "segments",
    "stream_text",
    "write_dataset",
]
