"""Training as an iterator, with the process hidden behind it.

A :class:`Run` is what the lora layer uses: iterate it for progress events, call
:meth:`Run.stop` from any thread, and read ``outcome`` when the iteration ends.
Three things that are not ours to invent are hidden behind it:

- **the child process** (:mod:`child`), which is where mlx-lm's Python API is
  actually called. It exists because training holds gigabytes of Metal memory,
  has to be interruptible, and must not be able to take Orange down with it.
- **the wire format** (:mod:`protocol`): one event per line on the child's stdout,
  everything else being mlx-lm's human log.
- **the stop**, which has to be a signal. ``stop()`` sets a flag *and* terminates
  the child's process group, which is what also wakes the reader parked on its
  stdout - and a timer escalates to SIGKILL if the child ignores the signal, so a
  stuck Metal call cannot leave the reader parked for ever. The blocking half is
  here too: iteration does not end, and the directory is not safe to read, until
  the child is gone. mlx-lm saves a checkpoint straight to its final path, so a
  directory read while a writer is alive can pick up a truncated file.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from .protocol import Event, decode

_CHILD = "orangecontrib.ai_suite.kernel.lora.mlx.child"

#: How long a terminated child may take to die before it is killed outright.
STOP_TIMEOUT = 10.0

#: How much of the child's human log is kept for a failure message.
_TAIL_LINES = 200


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
        super().__init__(f"mlx_lm training failed ({returncode}):\n{''.join(tail)}")


def command(spec: TrainingSpec) -> list[str]:
    """The one way to start a run: our child module, and the run as one JSON argument."""
    return [sys.executable, "-m", _CHILD, json.dumps(asdict(spec))]


class Run:
    """One training run: iterate it for events; ``stop()`` from any thread.

    Iteration is what spawns the child, so constructing a run costs nothing and a
    run that is stopped before it starts never spawns one. When iteration ends -
    normally, by :meth:`stop`, or because the consumer left the loop - the child
    is gone and ``outcome`` says whether a stop cut the run short.
    """

    def __init__(self, spec: TrainingSpec, *, stop_timeout: float = STOP_TIMEOUT) -> None:
        self.spec = spec
        self.outcome: Outcome | None = None
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self._stop_timeout = stop_timeout
        self._reaping = threading.Lock()

    # ---- control ----------------------------------------------------------

    @property
    def stopped(self) -> bool:
        """Whether a stop has been asked for (it may not have happened yet)."""
        return self._stop.is_set()

    @property
    def tail(self) -> list[str]:
        """The end of the child's human log, for a failure message."""
        return list(self._tail)

    def stop(self) -> None:
        """Ask the run to stop; returns at once, so any thread may call it.

        Sets the flag the reader checks, and terminates the child's process group
        - which is also what wakes the reader, since the signal closes the pipe it
        is parked on. If the child ignores it (a Metal call that will not be
        interrupted), a timer escalates to SIGKILL; the *blocking* wait for the
        child belongs to iteration and to :meth:`close`, never here.
        """
        self._stop.set()
        process = self._process
        if process is None or process.poll() is not None:
            return
        _terminate(process)
        escalation = threading.Timer(self._stop_timeout, _kill, args=(process,))
        escalation.daemon = True
        escalation.start()

    def close(self) -> None:
        """Stop the run and wait for its child: for a caller that leaves early."""
        self.stop()
        process = self._process
        if process is not None:
            self._reap(process)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    # ---- the events -------------------------------------------------------

    def __iter__(self) -> Iterator[Event]:
        if self.outcome is not None:
            raise RuntimeError("a run reports its events once")
        if self.stopped:
            self.outcome = Outcome(stopped=True)
            return
        process = subprocess.Popen(
            command(self.spec),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            start_new_session=True,  # one signal reaches the whole tree
        )
        self._process = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                event = decode(line)
                if event is None:
                    self._tail.append(line)
                    continue
                yield event
            process.wait()  # end of output: collect the child's status
            if not self._expected(process.returncode):
                raise TrainingFailed(process.returncode, self.tail)
        finally:
            # Nothing outside may read the adapter directory while the child can
            # still write to it.
            self._reap(process)
            self.outcome = Outcome(stopped=self.stopped)

    # ---- the process ------------------------------------------------------

    def _expected(self, returncode: int) -> bool:
        """Whether this exit status is a normal end: clean, or a stop we asked for."""
        if returncode == 0:
            return True
        if returncode == -signal.SIGTERM:
            return True
        return self.stopped and returncode == -signal.SIGKILL

    def _reap(self, process: subprocess.Popen[str]) -> None:
        """Make sure the child is gone before anything reads its directory."""
        with self._reaping:
            if process.poll() is not None:
                process.wait()
                return
            _terminate(process)
            try:
                process.wait(timeout=self._stop_timeout)
            except subprocess.TimeoutExpired:
                _kill(process)
                process.wait()


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
