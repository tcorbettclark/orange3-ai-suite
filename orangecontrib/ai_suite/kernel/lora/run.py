"""One run: its specification, its scratch directory, and what a stop leaves behind.

The run is :class:`~.mlx_process.Run`'s: this module builds its
:class:`~.mlx_spec.TrainingSpec` from the caller's options, stages the rows into a
directory it owns, and adopts whatever the child left there as a
:class:`~.artifact.LoRAAdapter`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import asdict, fields
from pathlib import Path

from . import mlx_layout, mlx_process, mlx_protocol, mlx_spec
from .artifact import LoRAAdapter
from .data import Format, LoRAData, LoRAOptions, split_for_training

#: Name of one run's directory. In the shared temp root the name also says whose
#: directory it is; inside a directory the caller named, only uniqueness is left
#: to say, so the run directory is just a run.
_SHARED_PREFIX = "orange-ai-lora-"
_RUN_PREFIX = "run-"


class TrainingStopped(RuntimeError):
    """The run was stopped before there was anything to adopt as an adapter."""


#: Every LoRAOptions field except the split's, which the child never sees: derived,
#: so a new setting is handed to the child automatically - and the child refuses a
#: setting it does not know, so it cannot be dropped on the floor.
_CHILD_OPTION_FIELDS = tuple(
    field.name for field in fields(LoRAOptions) if field.name != "validation_fraction"
)


def _run_spec(
    base_model: str | Path,
    data_dir: str | Path,
    adapter_dir: str | Path,
    options: LoRAOptions,
    format: Format,
) -> mlx_spec.TrainingSpec:
    """The run's spec in mlx-lm's names, with prompt masking set from the data format.

    SFT (``"pairs"``) masks the prompt, because the prompt is the caller's text rather
    than a target.
    """
    return mlx_spec.TrainingSpec(
        model=str(base_model),
        data=str(data_dir),
        adapter_path=str(adapter_dir),
        mask_prompt=format == "pairs",
        **{name: getattr(options, name) for name in _CHILD_OPTION_FIELDS},
    )


def _check_workdir(workdir: Path | None) -> Path | None:
    """The workdir as a Path, or None; a path that exists and is not a directory raises.

    The caller creates it or does not: the run creates the directory when it begins.
    """
    if workdir is None:
        return None
    workdir = Path(workdir)
    if workdir.exists() and not workdir.is_dir():
        raise ValueError(f"workdir must be a directory, got a file: {workdir}")
    return workdir


def _run_directory(workdir: Path | None) -> Path:
    """A fresh directory for one run, which the caller owns and deletes.

    ``workdir`` must be a directory nothing else writes in: every child of it is one
    run's scratch, so deleting it is the whole cleanup.
    """
    if workdir is None:
        return Path(tempfile.mkdtemp(prefix=_SHARED_PREFIX))
    workdir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=workdir))


class TrainingRun:
    """One LoRA run: iterate it once for progress, and ``stop()`` it from any thread.

    A bad setting raises when the run is asked for, not at the first event. The run
    writes into a fresh directory with ``data/`` and ``adapter/`` subdirectories, which
    the caller owns and must delete after the last :class:`LoRAModel` pointing into it:
    ``adapter.adapter_path`` points inside it and ``training["run_dir"]`` names it. A
    stop adopts the newest checkpoint and records its iteration in ``iters_completed``;
    a stop before the first checkpoint raises :class:`TrainingStopped`.
    """

    def __init__(
        self,
        base_model: str | Path,
        data: LoRAData,
        options: LoRAOptions,
        workdir: str | Path | None,
        name: str,
    ) -> None:
        self.base_model = base_model
        self.data = data
        self.options = options
        self.name = name
        self.adapter: LoRAAdapter | None = None  # set when the run ends well
        self.train, self.valid = split_for_training(data, options)  # eager: fail at the call
        self._workdir = _check_workdir(workdir)
        self._stop = threading.Event()
        self._run: mlx_process.Run | None = None
        self._iterated = False

    # ---- control ----------------------------------------------------------

    @property
    def stopped(self) -> bool:
        """Whether a stop has been asked for (it may not have happened yet)."""
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask the run to stop; returns at once, so any thread may call it."""
        self._stop.set()
        if self._run is not None:
            self._run.stop()

    def close(self) -> None:
        """Stop the run and wait for its child: for a caller that leaves early."""
        self.stop()
        if self._run is not None:
            self._run.close()

    def __enter__(self) -> TrainingRun:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    # ---- the events -------------------------------------------------------

    def __iter__(self) -> Iterator[mlx_protocol.Event]:
        if self._iterated:
            raise RuntimeError("a run is iterated once: ask for a new one")
        self._iterated = True
        if self.stopped:
            raise TrainingStopped("training was stopped before it started; nothing was adopted")
        run_dir = _run_directory(self._workdir)
        adapter_dir = run_dir / "adapter"
        spec = _run_spec(
            self.base_model,
            mlx_layout.write_dataset(run_dir / "data", self.data.format, self.train, self.valid),
            adapter_dir,
            self.options,
            self.data.format,
        )
        run = mlx_process.Run(spec)
        self._run = run
        try:
            try:
                yield from run
            except mlx_process.TrainingFailed as exc:
                raise RuntimeError(
                    f"mlx_lm lora failed ({exc.returncode}); the run directory "
                    f"{run_dir} keeps the staged data and log tail:\n{''.join(exc.tail[-20:])}"
                ) from exc
        finally:
            self._run = None
        self.adapter = self._adopt(run_dir, adapter_dir, run.outcome)

    # ---- the artifact -----------------------------------------------------

    def _adopt(
        self, run_dir: Path, adapter_dir: Path, outcome: mlx_process.Outcome | None
    ) -> LoRAAdapter:
        """Turn whatever the run left in the adapter directory into an artifact."""
        weights = mlx_layout.adapter_weights(adapter_dir)
        if weights is None:
            if outcome is not None and outcome.stopped:
                raise TrainingStopped(
                    "training was stopped before mlx-lm wrote a checkpoint "
                    "(save_every iterations must elapse first); nothing was adopted, and "
                    f"the run directory {run_dir} holds the log tail"
                )
            raise RuntimeError(f"training ended without adapter files; see {run_dir}")
        canonical = adapter_dir / mlx_layout.ADAPTER_WEIGHTS
        completed = self.options.iters
        if weights != canonical:
            completed = mlx_layout.checkpoint_iteration(weights)
            shutil.copy2(weights, canonical)
            # mlx-lm writes one config per run, at the start (lora.py: save_config),
            # and none beside the numbered checkpoints, so the run's config is the
            # only one there is and it stays: the LoRA structure it describes does
            # not change mid-run. How far the run got is training["iters_completed"].
        adapter_config = {}
        config_path = adapter_dir / mlx_layout.ADAPTER_CONFIG
        if config_path.exists():
            adapter_config = json.loads(config_path.read_text())
        return LoRAAdapter(
            base_model=str(self.base_model),
            adapter_path=str(adapter_dir),
            training={
                "options": asdict(self.options),
                "adapter_config": adapter_config,
                "iters_completed": completed,
                "stopped": completed < self.options.iters,
                "run_dir": str(run_dir),
            },
            name=self.name,
        )


def train_lora(
    base_model: str | Path,
    data: LoRAData,
    *,
    options: LoRAOptions | None = None,
    workdir: str | Path | None = None,
    name: str = "",
) -> TrainingRun:
    """A LoRA training run (QLoRA on a quantised base), returned but not started.

    The child process exists only while the run is iterated. ``name`` is the
    artifact's name, which only the caller can choose. See :class:`TrainingRun` for the
    run's directory, its stop and its artifact.
    """
    return TrainingRun(base_model, data, options or LoRAOptions(), workdir, name)
