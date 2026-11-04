"""The LoRA model object: an adapter travels with its base model.

An adapter is meaningless without the exact base model it was trained against,
so a :class:`LoRAAdapter` is a directory containing the adapter weights, its
config, and a ``lora_meta.json`` with the base-model identity and
hyperparameters. Loading the artifact reproduces the trained setup, and the
adapter on/off switch gives the before/after comparison on the identical base
model.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MLX_IMPORT_ERROR = (
    "MLX not available - orange3-nn-suite[lora] requires macOS on Apple Silicon "
    "(pip install orange3-nn-suite[lora])"
)

_ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
_META_FILE = "lora_meta.json"
_MAX_SEQ_LEN = 512


def _require_mlx() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MLX_IMPORT_ERROR) from exc


@dataclass
class LoRAAdapter:
    """The fine-tuning artifact: adapter files + base-model identity, one unit."""

    base_model: str  # path or HF repo id of the base model
    adapter_path: str | None  # None = the raw base model itself (no adapter)
    hyperparameters: dict
    name: str = ""

    def save(self, dest: str | Path) -> Path:
        """Write the self-contained artifact directory.

        Metadata and file copying do not require MLX; only ``LoRAModel`` does.
        """
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        if self.adapter_path is not None:
            src = Path(self.adapter_path)
            for fname in _ADAPTER_FILES:
                if (src / fname).exists():
                    shutil.copy2(src / fname, out / fname)
        (out / _META_FILE).write_text(
            json.dumps(
                {
                    "base_model": self.base_model,
                    "adapter_path": None,  # the copy is the adapter now
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
        src = Path(path)
        if (src / _META_FILE).exists():
            meta = json.loads((src / _META_FILE).read_text())
            has_weights = (src / "adapters.safetensors").exists()
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

    Exposes the kernel convention
    ``generate(prompt, temperature, max_new_tokens, seed)``. The
    ``adapter_enabled`` switch selects the raw base model or the
    adapter-fitted one: the identical model, before and after, with no
    confound. Both variants are loaded lazily and cached.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        hyperparameters: dict | None = None,
        name: str = "",
    ):
        _require_mlx()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter_enabled = adapter_path is not None
        self.hyperparameters = dict(hyperparameters or {})
        self.name = name
        self._models: dict[bool, Any] = {}  # variant -> (model, tokenizer)

    def _variant(self, with_adapter: bool):
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
        """Temperature sampling; temperature <= 0 means greedy.

        The sampling loop lives inside ``mlx_lm`` rather than in the kernel,
        but the convention surface is identical to the Keras models. mlx-lm
        exposes temperature through a sampler object rather than a kwarg.
        """
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
            sampler=make_sampler(temp=max(0.0, float(temperature))),
            verbose=False,
        )
        # mlx-lm returns the completion only; the kernel convention is the
        # full text (prompt echoed), matching the Keras LanguageModel.
        sep = " " if prompt and output and not prompt[-1:].isspace() and output[0].isalnum() else ""
        return prompt + sep + output

    def perplexity(self, text: str, max_tokens: int = 4000) -> dict:
        """Exponential mean cross-entropy over the text's next-token predictions.

        Returns ``{'loss', 'perplexity'}`` on the same convention as the Keras
        ``LanguageModel.perplexity``, for in-domain / out-of-domain tables.
        Chunks long text into non-overlapping windows.
        """
        import math

        import mlx.core as mx
        import mlx.nn as nn

        model, tokenizer = self._variant(self.adapter_enabled)
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            return {"loss": float("nan"), "perplexity": float("nan")}
        nll_sum, n_tokens = 0.0, 0
        step = _MAX_SEQ_LEN - 1  # no overlap between windows
        limit = min(len(ids), max_tokens)
        for start in range(0, limit - 1, step):
            window = ids[start : min(start + _MAX_SEQ_LEN, limit)]
            if len(window) < 2:
                continue
            x = mx.array([window[:-1]])
            targets = mx.array(window[1:])
            logits = model(x)
            log_probs = nn.log_softmax(logits.reshape(logits.shape[1], -1), axis=-1)
            picked = log_probs[mx.arange(len(window) - 1), targets]
            nll_sum += float(-picked.sum())
            n_tokens += len(window) - 1
        loss = nll_sum / max(n_tokens, 1)
        return {"loss": float(loss), "perplexity": float(math.exp(loss))}

    def artifact(self) -> LoRAAdapter:
        return LoRAAdapter(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            hyperparameters=self.hyperparameters,
            name=self.name,
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> LoRAModel:
        adapter = LoRAAdapter.load(path)
        return cls(
            adapter.base_model,
            adapter.adapter_path,
            adapter.hyperparameters,
            adapter.name,
        )

    def save(self, dest: str | Path) -> Path:
        return self.artifact().save(dest)
