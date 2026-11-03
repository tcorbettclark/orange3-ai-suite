"""The LoRA model object: adapter travels with its base model.

The artifact convention (§4.3): an adapter is meaningless without the exact
base model it was trained against — "adapter serialized with base model" as
one unit. A `LoRAAdapter` is a directory containing the adapter weights, its
config, and a meta.json carrying the base-model identity + hyperparameters;
loading it reproduces the *trained setup*, and the on/off switch gives the
Stage-2/Stage-3 before/after comparison on the identical base model.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MLX_IMPORT_ERROR = (
    "MLX not available — nn-kernel[lora] requires macOS on Apple "
    "Silicon (pip install orange3-nn-suite[lora])"
)

_ADAPTORS = ("adapters.safetensors", "adapter_config.json")
_META_FILE = "lora_meta.json"


def _require_mlx():
    try:
        from mlx_lm import generate, load  # noqa: F401
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
        """Write the self-contained artifact directory."""
        _require_mlx()
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        if self.adapter_path is not None:
            src = Path(self.adapter_path)
            for fname in _ADAPTORS:
                if (src / fname).exists():
                    shutil.copy2(src / fname, out / fname)
        (out / _META_FILE).write_text(json.dumps({
            "base_model": self.base_model,
            "adapter_path": None,  # the copy *is* the adapter now
            "hyperparameters": self.hyperparameters,
            "name": self.name,
            "saved_epoch": time.strftime("%Y-%m-%d %H:%M"),
        }, indent=1))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "LoRAAdapter":
        _require_mlx()  # also validates platform before touching the artifact
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
        return (f"LoRAAdapter(base={self.base_model}, "
                f"adapter={'yes' if self.adapter_path else 'off'})")


class LoRAModel:
    """Stage-2/3 model: the base model with an optional adapter, exposing the
    kernel convention ``generate(prompt, temperature, max_new_tokens, seed)``.

    ``adapter_enabled`` switches between the raw base model and the
    adapter-fitted one — the identical model, before/after, no confound.
    Both variants are loaded lazily and cached (1.5 GB 4-bit each).
    """

    def __init__(self, base_model: str, adapter_path: str | None = None):
        _require_mlx()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter_enabled = adapter_path is not None
        self._models: dict[bool, Any] = {}  # variant -> (model, tokenizer)

    def _variant(self, with_adapter: bool):
        from mlx_lm.utils import load

        if with_adapter not in self._models:
            self._models[with_adapter] = load(
                self.base_model,
                adapter_path=self.adapter_path if with_adapter else None,
            )
        return self._models[with_adapter]

    @property
    def tokenizer(self):
        """Travels with the model (kernel convention; shared by variants)."""
        return self._variant(self.adapter_enabled)[1]

    def generate(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> str:
        """Kernel sampling convention; temperature <= 0 = greedy.

        The MLX generation loop lives inside mlx_lm (unlike the Keras models,
        where the ~20-line loop is kernel-owned); the convention surface is
        identical either way. mlx-lm 0.32 exposes temperature through a
        sampler object rather than a kwarg.
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
        # full text (prompt echoed — matching the Keras LanguageModel)
        sep = (
            " "
            if prompt and output and not prompt[-1:].isspace()
            and output[0].isalnum()
            else ""
        )
        return prompt + sep + output

    def perplexity(self, text: str, max_tokens: int = 4000, batch: int = 8) -> dict:
        """Exponential mean cross-entropy over the text's next-token predictions
        — the same convention as the Keras LanguageModel.perplexity, feeding
        the spec §6 in-domain / out-of-domain tables. Chunks by ``batch``
        windows of ``max_seq_length``; returns {'loss', 'perplexity'}.
        """
        import math

        import mlx.nn as nn
        import mlx.core as mx

        model, tokenizer = self._variant(self.adapter_enabled)
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            return {"loss": float("nan"), "perplexity": float("nan")}
        nll_sum, n_tokens = 0.0, 0
        step = self._seq_len() - batch
        step = max(step, 1)
        for start in range(0, min(len(ids), max_tokens) - 1, step):
            window = ids[start : start + self._seq_len()]
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

    def _seq_len(self) -> int:
        # inference window; short windows keep the 32000-vocab logits small
        return 512

    def artifact(self) -> LoRAAdapter:
        return LoRAAdapter(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            hyperparameters={},
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> "LoRAModel":
        adapter = LoRAAdapter.load(path)
        return cls(adapter.base_model, adapter.adapter_path)

    def save(self, dest: str | Path) -> Path:
        return self.artifact().save(dest)