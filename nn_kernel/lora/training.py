"""LoRA training: the two data formats and the streaming run.

Two data formats teach two distinct lessons through one mechanism:

1. **Raw domain corpus -> continued pre-training.** Self-supervised
   next-token loss; rows are ``{"text": ...}``.
2. **Prompt-response pairs -> supervised fine-tuning.** The supervision lives
   in the data format, not in the loss: a human authored the prompt and the
   completion. Rows are ``{"prompt": ..., "completion": ...}`` and mlx-lm
   masks the prompt tokens, so loss is computed on the completion only.

Training runs ``python -m mlx_lm lora`` in a subprocess. Its stdout is parsed
into metric dicts for the widget's live loss curve, and ``stop_requested``
terminates the child process cleanly (the widget's Stop button).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .model import LoRAAdapter

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# training row: "  20    0.725 ▼    184    3.3k";  val row: "  50  val  0.018  1.40s"
_TRAIN_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▲▼]\s+(\d+)\s+([\dk.]+)\s*$")
_VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)\s+([\d.]+s)\s*$")


@dataclass
class LoraOptions:
    """Hyperparameters for a classroom-sized 4-bit QLoRA run."""

    iters: int = 300
    learning_rate: float = 1e-4
    num_layers: int = 16
    batch_size: int = 4
    max_seq_length: int = 512
    steps_per_report: int = 20
    steps_per_eval: int = 50
    seed: int = 42
    fine_tune_type: str = "lora"


def _n_valid(n: int, valid_fraction: float) -> int:
    """Validation rows for a fraction in [0, 1); always leave at least one train row."""
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    if valid_fraction == 0.0 or n < 2:
        return 0
    return min(max(1, round(n * valid_fraction)), n - 1)


@dataclass
class LoraData:
    """One of the two teaching formats, staged as mlx-lm jsonl files."""

    kind: str  # "text" (continued pre-training) | "pairs" (SFT, masked prompts)
    train: list[str | tuple]
    valid: list[str | tuple] = field(default_factory=list)

    @classmethod
    def from_texts(cls, texts: Iterable[str], valid_fraction: float = 0.1):
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 2:
            raise ValueError("need at least 2 texts (train + valid)")
        n_valid = _n_valid(len(texts), valid_fraction)
        return cls("text", texts[: len(texts) - n_valid], texts[len(texts) - n_valid :])

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], valid_fraction: float = 0.1):
        pairs = [(str(p), str(c)) for p, c in pairs if p.strip() and c.strip()]
        if len(pairs) < 4:
            raise ValueError("need at least 4 prompt-completion pairs (SFT)")
        n_valid = _n_valid(len(pairs), valid_fraction)
        return cls("pairs", pairs[: len(pairs) - n_valid], pairs[len(pairs) - n_valid :])

    def stage(self, directory: str | Path) -> Path:
        """Write {train, valid}.jsonl in the format mlx-lm accepts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for split, rows in (("train", self.train), ("valid", self.valid)):
            path = directory / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    if self.kind == "text":
                        f.write(json.dumps({"text": row}) + "\n")
                    else:
                        prompt, completion = row
                        f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
        return directory


def stream_metrics(line: str) -> dict | None:
    """Parse one mlx-lm training-log line into a metric dict (or None)."""
    line = _ANSI.sub("", line.rstrip("\n"))
    m = _TRAIN_ROW.match(line)
    if m:
        return {
            "iter": int(m.group(1)),
            "loss": float(m.group(2)),
            "tok_s": int(m.group(3)),
            "kind": "train",
        }
    m = _VAL_ROW.match(line)
    if m:
        return {"iter": int(m.group(1)), "loss": float(m.group(2)), "kind": "val"}
    return None


def train_lora(
    base_model: str | Path,
    data: LoraData,
    adapter_dir: str | Path,
    options: LoraOptions | None = None,
    on_metric: Callable[[dict], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    data_dir: str | Path | None = None,
) -> LoRAAdapter:
    """Run QLoRA training; stream metrics; return the adapter artifact.

    The subprocess runs in its own process group, so stopping terminates the
    whole child tree; the adapter files are verified afterwards.
    """
    opts = options or LoraOptions()
    if len(data.valid) < opts.batch_size and opts.iters > 0:
        raise ValueError(
            f"valid split has {len(data.valid)} rows but batch_size is "
            f"{opts.batch_size} - mlx-lm requires the valid split to hold at "
            "least one full batch; add data or reduce batch_size"
        )
    staged = data.stage(data_dir or Path(adapter_dir).with_suffix(".data"))
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(base_model),
        "--train",
        "--data",
        str(staged),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(opts.iters),
        "--learning-rate",
        str(opts.learning_rate),
        "--num-layers",
        str(opts.num_layers),
        "--batch-size",
        str(opts.batch_size),
        "--max-seq-length",
        str(opts.max_seq_length),
        "--steps-per-report",
        str(opts.steps_per_report),
        "--steps-per-eval",
        str(opts.steps_per_eval),
        "--seed",
        str(opts.seed),
        "--fine-tune-type",
        opts.fine_tune_type,
    ]

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
    try:
        reader = threading.Thread(
            target=_pump, args=(process, tail, on_metric, on_line), daemon=True
        )
        reader.start()
        while process.poll() is None:
            if stop_requested is not None and stop_requested():
                os.killpg(process.pid, signal.SIGTERM)
                break
            threading.Event().wait(0.2)
        reader.join(timeout=10)
        if process.returncode not in (0, None, -signal.SIGTERM):
            tail_text = "".join(tail[-20:])
            raise RuntimeError(f"mlx_lm lora failed ({process.returncode}):\n{tail_text}")
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    if not (adapter_dir / "adapters.safetensors").exists():
        raise RuntimeError("training ended without adapter files")
    adapter_config = {}
    if (adapter_dir / "adapter_config.json").exists():
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    return LoRAAdapter(
        base_model=str(base_model),
        adapter_path=str(adapter_dir),
        hyperparameters={
            **{
                k: getattr(opts, k)
                for k in (
                    "iters",
                    "learning_rate",
                    "num_layers",
                    "batch_size",
                    "max_seq_length",
                    "seed",
                    "fine_tune_type",
                )
            },
            "adapter_config": adapter_config,
        },
        name=adapter_dir.name,
    )


def _pump(process, tail: list[str], on_metric, on_line) -> None:
    """Stream the child's stdout; keep the last lines for error reporting."""
    for line in process.stdout:  # type: ignore[union-attr]
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
        if on_line is not None:
            on_line(line)
        if on_metric is not None:
            metric = stream_metrics(line)
            if metric is not None:
                on_metric(metric)
