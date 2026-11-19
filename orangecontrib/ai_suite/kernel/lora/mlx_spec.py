"""One training run as the child receives it: every setting a training call takes.

The parent writes a :class:`TrainingSpec` as JSON on the child's command line and the
child reads it back by name, so a field one side does not know fails there rather than
silently. Its counterpart is :mod:`mlx_protocol`, which carries the child's reports
back the other way.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSpec:
    """One run, as mlx-lm describes it: every setting a training call takes."""

    model: str
    data: str
    adapter_path: str
    iters: int
    learning_rate: float
    num_layers: int
    batch_size: int
    max_seq_length: int
    steps_per_report: int
    steps_per_eval: int
    save_every: int
    seed: int
    mask_prompt: bool
