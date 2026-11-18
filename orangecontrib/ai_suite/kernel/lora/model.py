"""The LoRA model object: an adapter travels with its base model.

An adapter is meaningless without the exact base model it was trained against,
so a :class:`LoRAAdapter` is a directory containing the adapter weights, its
config, and a ``lora_meta.json`` with the base-model identity and
hyperparameters. Loading the artifact reproduces the trained setup.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MLX_IMPORT_ERROR = (
    "MLX not available - orange3-ai-suite[lora] requires macOS on Apple Silicon "
    "(pip install orange3-ai-suite[lora])"
)

_ADAPTER_WEIGHTS = "adapters.safetensors"
_ADAPTER_CONFIG = "adapter_config.json"
_META_FILE = "lora_meta.json"
_EVAL_WINDOW = 512  # ids per perplexity window; fixed so runs stay comparable


def _require_mlx() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MLX_IMPORT_ERROR) from exc


def is_4bit(base_model: str | Path) -> bool:
    """Whether a local model directory declares 4-bit quantisation.

    Reads ``quantization.bits`` from the model's ``config.json``. A Hugging Face
    id is not downloaded to check, so it counts as not 4-bit.
    """
    config = Path(base_model) / "config.json"
    if not config.is_file():
        return False
    quantization = json.loads(config.read_text()).get("quantization") or {}
    return quantization.get("bits") == 4


def _segments(ids: list[int], max_tokens: int, window: int) -> list[list[int]]:
    """Windows of at most ``window`` inputs, plus the target that follows each.

    Consecutive windows share one id: the last target of one window is the final
    input of the next, so every id after the first is scored exactly once. Only
    the first ``max_tokens`` ids are used.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    limit = min(len(ids), max_tokens)
    step = window - 1
    return [ids[start : min(start + window, limit)] for start in range(0, limit - 1, step)]


@dataclass
class LoRAAdapter:
    """The fine-tuning artifact: adapter files + base-model identity, one unit."""

    base_model: str  # path or HF repo id of the base model
    adapter_path: str | None  # None = the raw base model itself (no adapter)
    hyperparameters: dict[str, Any]
    name: str = ""

    def save(self, dest: str | Path) -> Path:
        """Write the self-contained artifact directory.

        Metadata and file copying do not require MLX; only ``LoRAModel`` does.
        The weights are checked for before anything is written, so an adapter
        path with no weights raises instead of silently saving a base model.
        """
        src = Path(self.adapter_path) if self.adapter_path is not None else None
        if src is not None and not (src / _ADAPTER_WEIGHTS).is_file():
            raise FileNotFoundError(f"no {_ADAPTER_WEIGHTS} in {src}")
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        if src is not None:
            for fname in (_ADAPTER_WEIGHTS, _ADAPTER_CONFIG):
                if (src / fname).exists():
                    shutil.copy2(src / fname, out / fname)
        (out / _META_FILE).write_text(
            json.dumps(
                {
                    "base_model": self.base_model,
                    "hyperparameters": self.hyperparameters,
                    "name": self.name,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M"),
                },
                indent=1,
            )
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> LoRAAdapter:
        """Read an adapter directory, or a raw model directory as the base model.

        An artifact carries ``lora_meta.json``; a directory without it is a plain
        MLX model directory, loaded with the adapter off.
        """
        src = Path(path)
        if (src / _META_FILE).exists():
            meta = json.loads((src / _META_FILE).read_text())
            has_weights = (src / _ADAPTER_WEIGHTS).exists()
            return cls(
                base_model=meta["base_model"],
                adapter_path=str(src) if has_weights else None,
                hyperparameters=meta.get("hyperparameters", {}),
                name=meta.get("name", ""),
            )
        return cls(base_model=str(src), adapter_path=None, hyperparameters={})

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"LoRAAdapter(base={self.base_model}, adapter={'yes' if self.adapter_path else 'off'})"
        )


class LoRAModel:
    """A base model with an optional LoRA adapter.

    Exposes the kernel conventions
    ``generate(prompt, temperature, max_new_tokens, seed)`` and ``evaluate(text)``.
    The ``adapter_enabled`` switch selects the raw base model or the
    adapter-fitted one: the identical model, before and after, with no
    confound. Both variants are loaded lazily and cached.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        name: str = "",
    ):
        _require_mlx()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter_enabled = adapter_path is not None
        self.hyperparameters = dict(hyperparameters or {})
        self.name = name
        self._models: dict[bool, tuple[Any, Any]] = {}  # variant -> (model, tokenizer)

    def _variant(self, with_adapter: bool) -> tuple[Any, Any]:
        from mlx_lm import load

        if with_adapter not in self._models:
            self._models[with_adapter] = load(
                self.base_model,
                adapter_path=self.adapter_path if with_adapter else None,
            )
        return self._models[with_adapter]

    @property
    def tokenizer(self):
        """Travels with the model (kernel convention; shared by both variants)."""
        return self._variant(self.adapter_enabled)[1]

    def generate(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> str:
        """Temperature sampling; 0 means greedy, and negative raises.

        The sampling loop lives inside ``mlx_lm`` rather than in the kernel,
        but the convention surface is identical to the Keras models. mlx-lm
        exposes temperature through a sampler object rather than a kwarg.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        from mlx.core import random as mx_random
        from mlx_lm.generate import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._variant(self.adapter_enabled)
        if seed is not None:
            mx_random.seed(int(seed))
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=int(max_new_tokens),
            sampler=make_sampler(temp=float(temperature)),
            verbose=False,
        )
        # mlx-lm returns the completion only; the kernel convention is the full
        # text (prompt echoed), matching the Keras LanguageModel. mlx-lm does not
        # say whether the completion begins with a space, so one is inserted
        # only when both sides are word characters.
        sep = " " if prompt and output and not prompt[-1:].isspace() and output[0].isalnum() else ""
        return prompt + sep + output

    def evaluate(self, text: str, max_tokens: int = 4000) -> dict:
        """Loss and perplexity over ``text``, in fixed 512-token windows.

        Returns ``{'loss', 'perplexity'}`` on the same convention as the Keras
        ``LanguageModel.evaluate``, for in-domain / out-of-domain tables. Only
        the first ``max_tokens`` ids are scored; fewer than two ids is undefined
        and returns NaN. ``adapter_enabled`` applies here too: turn it off to
        score the raw base model.
        """
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        import mlx.core as mx
        import mlx.nn as nn

        model, tokenizer = self._variant(self.adapter_enabled)
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            return {"loss": float("nan"), "perplexity": float("nan")}
        nll_sum, n_tokens = 0.0, 0
        for segment in _segments(ids, max_tokens, _EVAL_WINDOW):
            x = mx.array([segment[:-1]])
            targets = mx.array(segment[1:])
            logits = model(x)
            log_probs = nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
            picked = log_probs[mx.arange(len(segment) - 1), targets]
            nll_sum += float(-picked.sum())
            n_tokens += len(segment) - 1
        loss = nll_sum / max(n_tokens, 1)
        return {"loss": float(loss), "perplexity": float(math.exp(loss))}

    def artifact(self) -> LoRAAdapter:
        """The model's adapter as a standalone artifact, ready to save."""
        return LoRAAdapter(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            hyperparameters=self.hyperparameters,
            name=self.name,
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> LoRAModel:
        """Load a model saved by :meth:`save`."""
        adapter = LoRAAdapter.load(path)
        return cls(
            adapter.base_model,
            adapter.adapter_path,
            adapter.hyperparameters,
            adapter.name,
        )

    def save(self, dest: str | Path) -> Path:
        return self.artifact().save(dest)
