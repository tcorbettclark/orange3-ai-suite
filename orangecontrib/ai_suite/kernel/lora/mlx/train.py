"""Training as an in-process-looking call, with the process hidden behind it.

:func:`run_training` is what the lora layer uses: a :class:`TrainingSpec` in,
per-iteration metrics out through a callback, a stop predicate, and a return when
the run is over. Three things that are not ours to invent are hidden behind it:

- **the child process** (:mod:`child`), which is where mlx-lm's Python API is
  actually called. It exists because training holds gigabytes of Metal memory,
  has to be interruptible, and must not be able to take Orange down with it.
- **the wire format** (:mod:`protocol`): one JSON metric per line on the child's
  stdout, everything else being mlx-lm's human log.
- **the stop**, which has to be a signal: mlx-lm's train loop cannot be
  cancelled, so the run asks the child's process group to stop, then insists, and
  only returns once the child is gone - because mlx-lm saves a checkpoint straight
  to its final path, and a directory read while a writer is alive can pick up a
  truncated file and adopt it as the run's artifact.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .protocol import decode

_CHILD = "orangecontrib.ai_suite.kernel.lora.mlx.child"

#: How long a stopped child may take to die before it is killed outright.
_STOP_TIMEOUT = 10


@dataclass(frozen=True)
class TrainingSpec:
    """One run, as mlx-lm describes it: every setting a training call takes.

    The lora layer builds this from its own options, so the mapping from a
    teaching setting to a library setting is written once, in one direction. A
    field added here and not there is a dataclass error rather than a silently
    ignored setting, because the child rebuilds the spec from JSON.
    """

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


@dataclass(frozen=True)
class Outcome:
    """What a finished run reports: whether a stop cut it short."""

    stopped: bool


class TrainingFailed(RuntimeError):
    """The child exited non-zero; ``tail`` is the end of its log."""

    def __init__(self, returncode: int, tail: list[str]) -> None:
        self.returncode = returncode
        self.tail = tail
        super().__init__(f"mlx_lm training failed ({returncode}):\n{''.join(tail[-20:])}")


def command(spec: TrainingSpec) -> list[str]:
    """The one way to start a run: our child module, and the run as one JSON argument."""
    return [sys.executable, "-m", _CHILD, json.dumps(asdict(spec))]


def run_training(
    spec: TrainingSpec,
    *,
    on_metric: Callable[[dict], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> Outcome:
    """Train one run; stream metrics; return when the child is gone.

    Returns :class:`Outcome` (whether a stop cut the run short) without judging
    what is on disk - whether a checkpoint is good enough to adopt is the lora
    layer's policy, since it is the layer that knows what a teaching run promises.
    Raises :class:`TrainingFailed` with the log tail when the child exits non-zero.
    """
    Path(spec.adapter_path).mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    process = subprocess.Popen(
        command(spec),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
        start_new_session=True,  # one signal reaches the whole tree
    )
    stopped = False
    reader: threading.Thread | None = None
    try:
        reader = threading.Thread(
            target=_pump, args=(process, tail, on_metric, on_line), daemon=True
        )
        reader.start()
        while process.poll() is None:
            if stop_requested is not None and stop_requested():
                stopped = True
                _terminate(process)
                break
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode not in (0, None, -signal.SIGTERM):
            raise TrainingFailed(process.returncode, tail)
    finally:
        # Nothing outside may read the adapter directory while the child can
        # still write to it.
        if process.poll() is None:
            _terminate(process)
            try:
                process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _kill(process)
                process.wait()
        if reader is not None:
            reader.join()
    return Outcome(stopped=stopped)


def _pump(
    process: subprocess.Popen[str],
    tail: list[str],
    on_metric: Callable[[dict], None] | None,
    on_line: Callable[[str], None] | None,
) -> None:
    """Stream the child's stdout: metrics to ``on_metric``, everything else as a log line."""
    assert process.stdout is not None
    for line in process.stdout:
        metric = decode(line)
        if metric is not None:
            if on_metric is not None:
                on_metric(metric)
            continue  # the data is consumed; the tail keeps the human log only
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
        if on_line is not None:
            on_line(line)


def _terminate(process: subprocess.Popen) -> None:
    """Ask the child's process group to stop; it may already be gone."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _kill(process: subprocess.Popen) -> None:
    """Stop the child's process group whether it agrees or not."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
