"""LoRA training: the two data formats, the split, and the streaming run.

Two data formats:

1. **Raw domain corpus -> continued pre-training.** Self-supervised next-token
   loss; rows are ``{"text": ...}``.
2. **Prompt-response pairs -> supervised fine-tuning.** A human authored the
   prompt and the completion. Rows are ``{"prompt": ..., "completion": ...}``;
   mlx-lm's prompt masking keeps the prompt tokens out of the loss, so only the
   completion is trained.

The run itself is :func:`.mlx.run_training`'s business: this module builds the
specification, decides what a stopped run should leave behind, and keeps the run's
scratch directory. mlx-lm writes a checkpoint every ``save_every`` iterations and
the final adapter after the last iteration, so a stop adopts the last checkpoint
and records how far the run got.

The train/validation split is a run setting, not a property of the rows: it
comes from :class:`LoRAOptions`, with the run's seed, so the two cannot
disagree.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from .._holdout import held_out_positions
from . import mlx
from .mlx import TrainingSpec
from .model import LoRAAdapter

Format = Literal["text", "pairs"]
Rows = list[str] | list[tuple[str, str]]

#: Name of one run's directory. In the shared temp root the name also says whose
#: directory it is; inside a directory the caller named, only uniqueness is left
#: to say, so the run directory is just a run.
_SHARED_PREFIX = "orange-ai-lora-"
_RUN_PREFIX = "run-"


class TrainingStopped(RuntimeError):
    """Training was stopped before mlx-lm wrote the adapter."""


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

        The stride is shared with lm's unit split (see :mod:`.._holdout`); this
        method owns the row-level policy: ``0`` means no validation at all, and
        at least one row stays on each side. A row's side therefore depends on
        where it sits in the corpus, not on its content, so identical rows may
        land on different sides and reordering the rows can move one across.
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
    """The train and validation rows, refusing a validation batch mlx-lm cannot use.

    A validation set smaller than one batch makes mlx-lm fail before training;
    raising here lets a caller reject the setup before starting a worker. An
    empty validation split (``validation_fraction=0``) is allowed: mlx-lm then
    trains without validation.
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
) -> TrainingSpec:
    """This run in mlx-lm's vocabulary: the one place the two sets of names meet.

    Paths are resolved here, and prompt masking follows the data format rather
    than a caller's flag: SFT masks the prompt because the prompt is the caller's
    text, not a target.
    """
    return TrainingSpec(
        model=str(base_model),
        data=str(data_dir),
        adapter_path=str(adapter_dir),
        mask_prompt=format == "pairs",
        **{name: getattr(options, name) for name in _CHILD_OPTION_FIELDS},
    )


def _run_directory(workdir: Path | None) -> Path:
    """A fresh directory for one run: the caller's to delete, never reused.

    Each run gets its own child, even inside a workdir, so a re-run can never
    meet a previous run's checkpoint - the failure this whole scheme exists to
    prevent. ``workdir`` must be a directory the caller owns exclusively:
    deleting it is how the caller cleans up, which is only safe if nothing else
    writes there.
    """
    if workdir is None:
        return Path(tempfile.mkdtemp(prefix=_SHARED_PREFIX))
    workdir = Path(workdir)
    if workdir.exists() and not workdir.is_dir():
        raise ValueError(f"workdir must be a directory, got a file: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=workdir))


def train_lora(
    base_model: str | Path,
    data: LoRAData,
    *,
    options: LoRAOptions | None = None,
    on_metric: Callable[[dict], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    workdir: str | Path | None = None,
    name: str = "",
) -> LoRAAdapter:
    """Run LoRA training (QLoRA on a quantised base); stream metrics; return the artifact.

    The split is ``options.validation_fraction`` of the rows, taken as a stride
    from ``options.seed``. The child runs in its own process group, so stopping
    terminates it and anything it started. mlx-lm writes a checkpoint every
    ``options.save_every`` iterations; if the run is stopped, the newest
    checkpoint becomes the adapter and ``iters_completed`` records the iteration
    it reached. A stop before the first checkpoint raises :class:`TrainingStopped`.

    The run owns all of its scratch: the staged "train"/"valid" rows and the
    adapter live in one fresh run-scoped directory (two subdirectories, "data"
    and "adapter"), so no run can ever meet another run's files and a checkpoint
    can never be mistaken for a previous result.

    That directory is the caller's to delete, and where it goes is the caller's
    choice: ``workdir`` is a directory the caller owns *exclusively* - every
    child of it is one run's scratch, so deleting the workdir is the whole
    cleanup - and each run gets a fresh child inside it. Without ``workdir`` the
    run lands in the system temp root. Either way the returned
    :class:`LoRAAdapter.adapter_path` points inside that directory and
    ``training["run_dir"]`` names it, so a caller can find it again. Keep the
    directory alive for as long as a :class:`LoRAModel` points into it, and call
    :meth:`LoRAAdapter.save` to copy the artifact somewhere durable. On a
    failure the raised exception names the directory, so the log tail can be
    inspected first.

    ``name`` is the artifact's name, and only the caller can know it: a
    directory generated here is not a name to show a person.
    """
    opts = options or LoRAOptions()
    train, valid = split_for_training(data, opts)
    run_dir = _run_directory(workdir)
    adapter_dir = run_dir / "adapter"
    spec = _run_spec(
        base_model,
        mlx.write_dataset(run_dir / "data", data.format, train, valid),
        adapter_dir,
        opts,
        data.format,
    )
    try:
        outcome = mlx.run_training(
            spec, on_metric=on_metric, on_line=on_line, stop_requested=stop_requested
        )
    except mlx.TrainingFailed as exc:
        raise RuntimeError(
            f"mlx_lm lora failed ({exc.returncode}); the run directory {run_dir} "
            f"keeps the staged data and log tail:\n{''.join(exc.tail[-20:])}"
        ) from exc

    weights = mlx.adapter_weights(adapter_dir)
    if weights is None:
        if outcome.stopped:
            raise TrainingStopped(
                "training was stopped before mlx-lm wrote a checkpoint "
                "(save_every iterations must elapse first); nothing was adopted, and "
                f"the run directory {run_dir} holds the log tail"
            )
        raise RuntimeError(f"training ended without adapter files; see {run_dir}")
    canonical = adapter_dir / mlx.ADAPTER_WEIGHTS
    completed = opts.iters
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
        base_model=str(base_model),
        adapter_path=str(adapter_dir),
        training={
            "options": asdict(opts),
            "adapter_config": adapter_config,
            "iters_completed": completed,
            "stopped": completed < opts.iters,
            "run_dir": str(run_dir),
        },
        name=name,
    )
