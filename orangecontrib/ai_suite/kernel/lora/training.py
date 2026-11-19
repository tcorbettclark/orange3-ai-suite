"""LoRA training: the data, the train/validation split, and the streaming run.

Two data formats: rows of raw text for continued pre-training, whose next-token loss
is self-supervised, and prompt/completion pairs for supervised fine-tuning, where
mlx-lm's prompt masking keeps the prompt tokens out of the loss. The run itself is :class:`~.mlx.Run`'s: this module
builds its specification, keeps its scratch directory and decides what a stopped run
leaves behind.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from .._holdout import held_out_positions
from . import mlx
from .model import LoRAAdapter

Format = Literal["text", "pairs"]
Rows = list[str] | list[tuple[str, str]]

#: Name of one run's directory. In the shared temp root the name also says whose
#: directory it is; inside a directory the caller named, only uniqueness is left
#: to say, so the run directory is just a run.
_SHARED_PREFIX = "orange-ai-lora-"
_RUN_PREFIX = "run-"


class TrainingStopped(RuntimeError):
    """The run was stopped before there was anything to adopt as an adapter."""


@dataclass(frozen=True)
class LoRAOptions:
    """Hyperparameters for one LoRA run on an MLX base model."""

    iters: int = 300
    learning_rate: float = 1e-4
    num_layers: int = 16
    batch_size: int = 4
    max_seq_length: int = 512
    validation_fraction: float = 0.1
    steps_per_report: int = 20
    steps_per_eval: int = 50
    save_every: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "iters",
            "num_layers",
            "batch_size",
            "steps_per_report",
            "steps_per_eval",
            "save_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be >= 2")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


#: Every LoRAOptions field except the split's, which never leaves this module:
#: derived, so a new setting is handed to the child automatically - and the child
#: refuses a setting it does not know, so it cannot be dropped on the floor.
_CHILD_OPTION_FIELDS = tuple(
    field.name for field in fields(LoRAOptions) if field.name != "validation_fraction"
)


@dataclass(frozen=True)
class LoRAData:
    """The rows of one of the two formats, before the train/validation split."""

    format: Format  # "text" (continued pre-training) | "pairs" (SFT)
    rows: Rows

    def __post_init__(self) -> None:
        if self.format not in ("text", "pairs"):
            raise ValueError(f"unknown format {self.format!r}: expected 'text' or 'pairs'")
        if self.format == "pairs":
            if not all(
                isinstance(row, tuple) and len(row) == 2 and all(isinstance(v, str) for v in row)
                for row in self.rows
            ):
                raise TypeError("pair rows must be (prompt, completion) string tuples")
        elif not all(isinstance(row, str) for row in self.rows):
            raise TypeError("text rows must be strings")

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> LoRAData:
        rows = [t for t in texts if t and t.strip()]
        if len(rows) < 2:
            raise ValueError("need at least 2 texts (train + validation)")
        return cls("text", rows)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> LoRAData:
        rows = [(str(p), str(c)) for p, c in pairs if str(p).strip() and str(c).strip()]
        if len(rows) < 4:
            raise ValueError("need at least 4 prompt-completion pairs (SFT)")
        return cls("pairs", rows)

    def split(self, valid_fraction: float, seed: int) -> tuple[Rows, Rows]:
        """Train and validation rows, chosen by position in the corpus.

        :func:`.._holdout.held_out_positions` picks the position; this method applies
        it to rows. ``0`` means no validation at all, and a corpus of two or more rows
        keeps at least one on each side. A row's side depends on where it sits, not on its
        content, so identical rows may land on different sides and reordering the rows
        can move one across.
        """
        if not 0.0 <= valid_fraction < 1.0:
            raise ValueError("valid_fraction must be in [0, 1)")
        if valid_fraction == 0.0:
            return list(self.rows), []
        held = set(held_out_positions(len(self.rows), valid_fraction, seed))
        train = [row for index, row in enumerate(self.rows) if index not in held]
        # sorted: the rows come back in corpus order, not in set order
        return train, [self.rows[index] for index in sorted(held)]


def split_for_training(data: LoRAData, options: LoRAOptions) -> tuple[Rows, Rows]:
    """The train and validation rows, refusing a validation split mlx-lm cannot use.

    A non-empty validation split holds at least ``batch_size`` rows: mlx-lm fails on
    anything less. An empty one is allowed, and mlx-lm then trains without validation.
    """
    train, valid = data.split(options.validation_fraction, options.seed)
    if valid and len(valid) < options.batch_size:
        rows = "row" if len(valid) == 1 else "rows"
        raise ValueError(
            f"valid split has {len(valid)} {rows} but batch_size is "
            f"{options.batch_size} - mlx-lm requires the valid split to hold at "
            "least one full batch; add data, reduce batch_size or raise "
            "validation_fraction"
        )
    return train, valid


def _run_spec(
    base_model: str | Path,
    data_dir: str | Path,
    adapter_dir: str | Path,
    options: LoRAOptions,
    format: Format,
) -> mlx.TrainingSpec:
    """The run's spec in mlx-lm's names, with prompt masking set from the data format.

    SFT (``"pairs"``) masks the prompt, because the prompt is the caller's text rather
    than a target.
    """
    return mlx.TrainingSpec(
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
        self._run: mlx.Run | None = None
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

    def __iter__(self) -> Iterator[mlx.Event]:
        if self._iterated:
            raise RuntimeError("a run is iterated once: ask for a new one")
        self._iterated = True
        if self.stopped:
            raise TrainingStopped("training was stopped before it started; nothing was adopted")
        run_dir = _run_directory(self._workdir)
        adapter_dir = run_dir / "adapter"
        spec = _run_spec(
            self.base_model,
            mlx.write_dataset(run_dir / "data", self.data.format, self.train, self.valid),
            adapter_dir,
            self.options,
            self.data.format,
        )
        run = mlx.Run(spec)
        self._run = run
        try:
            try:
                yield from run
            except mlx.TrainingFailed as exc:
                raise RuntimeError(
                    f"mlx_lm lora failed ({exc.returncode}); the run directory "
                    f"{run_dir} keeps the staged data and log tail:\n{''.join(exc.tail[-20:])}"
                ) from exc
        finally:
            self._run = None
        self.adapter = self._adopt(run_dir, adapter_dir, run.outcome)

    # ---- the artifact -----------------------------------------------------

    def _adopt(self, run_dir: Path, adapter_dir: Path, outcome: mlx.Outcome | None) -> LoRAAdapter:
        """Turn whatever the run left in the adapter directory into an artifact."""
        weights = mlx.adapter_weights(adapter_dir)
        if weights is None:
            if outcome is not None and outcome.stopped:
                raise TrainingStopped(
                    "training was stopped before mlx-lm wrote a checkpoint "
                    "(save_every iterations must elapse first); nothing was adopted, and "
                    f"the run directory {run_dir} holds the log tail"
                )
            raise RuntimeError(f"training ended without adapter files; see {run_dir}")
        canonical = adapter_dir / mlx.ADAPTER_WEIGHTS
        completed = self.options.iters
        if weights != canonical:
            completed = mlx.checkpoint_iteration(weights)
            shutil.copy2(weights, canonical)
            # mlx-lm writes one config per run, at the start (lora.py: save_config),
            # and none beside the numbered checkpoints, so the run's config is the
            # only one there is and it stays: the LoRA structure it describes does
            # not change mid-run. How far the run got is training["iters_completed"].
        adapter_config = {}
        config_path = adapter_dir / mlx.ADAPTER_CONFIG
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
