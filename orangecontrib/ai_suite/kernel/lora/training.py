"""LoRA training: the two data formats, the split, and the streaming run.

Two data formats:

1. **Raw domain corpus -> continued pre-training.** Self-supervised next-token
   loss; rows are ``{"text": ...}``.
2. **Prompt-response pairs -> supervised fine-tuning.** A human authored the
   prompt and the completion. Rows are ``{"prompt": ..., "completion": ...}``;
   ``--mask-prompt`` keeps the prompt tokens out of the loss, so only the
   completion is trained.

The run is ``python -m mlx_lm lora`` in a subprocess. Its stdout is parsed into
per-iteration metric dicts. ``stop_requested`` terminates the child process
tree; mlx-lm writes a checkpoint every ``save_every`` iterations and the final
adapter after the last iteration, so a stop adopts the last checkpoint and
records how far the run got.

The train/validation split is a run setting, not a property of the rows: it
comes from :class:`LoRAOptions`, with the run's seed, so the two cannot
disagree.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .._split import key_order
from .model import _ADAPTER_CONFIG, _ADAPTER_WEIGHTS, LoRAAdapter

Format = Literal["text", "pairs"]
Rows = list[str] | list[tuple[str, str]]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# training row: "  20    0.725 ▼    184    3.3k";  val row: "  50  val  0.018  1.40s"
_TRAIN_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▲▼]\s+([\d,]+)\s+([\dk.]+)\s*$")
_VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)\s+([\d.]+s)\s*$")
_TRAINABLE = re.compile(r"Trainable parameters:\s*([\d.]+)%\s*\(([^)]+)\)")


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
        """Train and validation rows, ordered by a salted hash of each row.

        The hashing is shared with lm's unit split (see :mod:`.._split`); this
        method owns the row-level policy: ``0`` means no validation at all,
        and at least one row stays on each side. A row's side depends on
        ``(seed, the row)``, not on the order the rows arrived in. Identical
        rows hash alike and may land in both splits.
        """
        if not 0.0 <= valid_fraction < 1.0:
            raise ValueError("valid_fraction must be in [0, 1)")
        if valid_fraction == 0.0:
            return list(self.rows), []
        n_valid = min(max(1, round(len(self.rows) * valid_fraction)), len(self.rows) - 1)
        ordered = [self.rows[i] for i in key_order([repr(row).encode() for row in self.rows], seed)]
        return ordered[n_valid:], ordered[:n_valid]


def _stage(format: Format, train: Rows, valid: Rows, directory: str | Path) -> Path:
    """Write ``{train, valid}.jsonl`` in the format mlx-lm accepts.

    ``valid.jsonl`` is omitted when there is no validation split: mlx-lm treats
    a missing file as no validation but fails on an empty one. A stale file from
    an earlier run is removed, so it cannot be picked up by mistake.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if valid:
        splits = [("train", train), ("valid", valid)]
    else:
        splits = [("train", train)]
        stale = directory / "valid.jsonl"
        if stale.exists():
            stale.unlink()
    for split, rows in splits:
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                record = (
                    {"text": row} if format == "text" else {"prompt": row[0], "completion": row[1]}
                )
                f.write(json.dumps(record) + "\n")
    return directory


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


def parse_metric(line: str) -> dict | None:
    """Parse one mlx-lm training-log line into a metric dict (or None)."""
    line = _ANSI.sub("", line.rstrip("\n"))
    m = _TRAINABLE.search(line)
    if m:
        return {
            "kind": "params",
            "pct": float(m.group(1)),
            "detail": m.group(2).strip(),
        }
    m = _TRAIN_ROW.match(line)
    if m:
        return {
            "iter": int(m.group(1)),
            "loss": float(m.group(2)),
            "tok_s": int(m.group(3).replace(",", "")),
            "kind": "train",
        }
    m = _VAL_ROW.match(line)
    if m:
        return {"iter": int(m.group(1)), "loss": float(m.group(2)), "kind": "val"}
    return None


def _lora_command(
    base_model: str | Path,
    data_dir: str | Path,
    adapter_dir: str | Path,
    options: LoRAOptions,
    format: Format,
) -> list[str]:
    """The mlx_lm lora invocation: one flag per option, prompt masking for SFT."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(base_model),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(options.iters),
        "--learning-rate",
        str(options.learning_rate),
        "--num-layers",
        str(options.num_layers),
        "--batch-size",
        str(options.batch_size),
        "--max-seq-length",
        str(options.max_seq_length),
        "--steps-per-report",
        str(options.steps_per_report),
        "--steps-per-eval",
        str(options.steps_per_eval),
        "--save-every",
        str(options.save_every),
        "--seed",
        str(options.seed),
        "--fine-tune-type",
        "lora",
    ]
    if format == "pairs":
        cmd.append("--mask-prompt")
    return cmd


def _terminate(process: subprocess.Popen) -> None:
    """Stop the child's process group; it may already be gone."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _adapter_weights(adapter_dir: Path) -> Path | None:
    """The final adapter if mlx-lm wrote it, else its newest numbered checkpoint."""
    canonical = adapter_dir / _ADAPTER_WEIGHTS
    if canonical.exists():
        return canonical
    checkpoints: list[tuple[int, Path]] = []
    for path in adapter_dir.glob(f"*_{_ADAPTER_WEIGHTS}"):
        try:
            checkpoints.append((int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def train_lora(
    base_model: str | Path,
    data: LoRAData,
    *,
    options: LoRAOptions | None = None,
    on_metric: Callable[[dict], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> LoRAAdapter:
    """Run LoRA training (QLoRA on a quantised base); stream metrics; return the artifact.

    The split is ``options.validation_fraction`` of the rows, ordered with
    ``options.seed``. The subprocess runs in its own process group, so stopping
    terminates the whole child tree. mlx-lm writes a checkpoint every
    ``options.save_every`` iterations; if the run is stopped, the newest
    checkpoint becomes the adapter and ``iters_completed`` records the iteration
    it reached. A stop before the first checkpoint raises :class:`TrainingStopped`.

    The run owns all of its scratch: the staged "train"/"valid" rows and the
    adapter live in one fresh run-scoped directory under the system temp root
    (two subdirectories, "data" and "adapter"), so no run can ever meet
    another run's files and a checkpoint can never be mistaken for a previous
    result. The returned :class:`LoRAAdapter.adapter_path` records where the
    adapter went and ``training["run_dir"]`` names the whole scratch directory;
    the caller owns cleanup - delete it once the artifact is saved with
    :meth:`LoRAAdapter.save` or discarded. On a failure the raised exception
    names the directory so the log tail can be inspected first.
    """
    opts = options or LoRAOptions()
    train, valid = split_for_training(data, opts)
    # One fresh scratch directory per run: a re-run can never meet an earlier
    # run's adapter, and a checkpoint can never be mistaken for its result.
    run_dir = Path(tempfile.mkdtemp(prefix="orange-ai-lora-"))
    adapter_dir = run_dir / "adapter"
    staged = _stage(data.format, train, valid, run_dir / "data")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = _lora_command(base_model, staged, adapter_dir, opts, data.format)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    tail: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    stopped = False
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
        reader.join(timeout=10)
        if process.returncode not in (0, None, -signal.SIGTERM):
            tail_text = "".join(tail[-20:])
            raise RuntimeError(
                f"mlx_lm lora failed ({process.returncode}); the run directory "
                f"{run_dir} keeps the staged data and log tail:\n{tail_text}"
            )
    finally:
        if process.poll() is None:
            _terminate(process)

    weights = _adapter_weights(adapter_dir)
    if weights is None:
        if stopped:
            raise TrainingStopped(
                "training was stopped before mlx-lm wrote a checkpoint "
                "(save_every iterations must elapse first); nothing was adopted, and "
                f"the run directory {run_dir} holds the log tail"
            )
        raise RuntimeError(f"training ended without adapter files; see {run_dir}")
    canonical = adapter_dir / _ADAPTER_WEIGHTS
    completed = opts.iters
    if weights != canonical:
        completed = int(weights.name.split("_", 1)[0])
        shutil.copy2(weights, canonical)
        # A checkpoint may carry its own config; adopt it with the weights so an
        # adopted mid-run adapter never travels without its matching config.
        prefix = weights.name[: -len(_ADAPTER_WEIGHTS)]
        paired_config = adapter_dir / f"{prefix}{_ADAPTER_CONFIG}"
        if paired_config.is_file():
            shutil.copy2(paired_config, adapter_dir / _ADAPTER_CONFIG)
    adapter_config = {}
    config_path = adapter_dir / _ADAPTER_CONFIG
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
        name=run_dir.name,
    )


def _pump(
    process: subprocess.Popen[str],
    tail: list[str],
    on_metric: Callable[[dict], None] | None,
    on_line: Callable[[str], None] | None,
) -> None:
    """Stream the child's stdout; keep the last lines for error reporting."""
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
        if on_line is not None:
            on_line(line)
        if on_metric is not None:
            metric = parse_metric(line)
            if metric is not None:
                on_metric(metric)
