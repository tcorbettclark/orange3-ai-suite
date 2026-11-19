"""The LoRA model object: an adapter travels with its base model.

An adapter is meaningless without the exact base model it was trained against,
so a :class:`LoRAAdapter` is a directory containing the adapter weights, its
config, and a ``lora_meta.json`` with the base-model identity and the training
run that produced it. Loading the artifact reproduces the trained setup.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import mlx

_META_FILE = "lora_meta.json"


def _echo(prompt: str, completion: str) -> str:
    """The prompt with the completion appended, one space between words.

    mlx-lm returns the completion only; the kernel convention is the full text
    (prompt echoed), matching the Keras ``LanguageModel``. mlx-lm does not say
    whether the completion begins with a space, so one is inserted only when the
    prompt ends in a non-space and the completion starts with an alphanumeric
    character.
    """
    sep = (
        " "
        if prompt and completion and not prompt[-1].isspace() and completion[0].isalnum()
        else ""
    )
    return prompt + sep + completion


@dataclass
class LoRAAdapter:
    """The fine-tuning artifact: adapter files + base-model identity, one unit."""

    base_model: str  # path or HF repo id of the base model
    adapter_path: str | None  # None = the raw base model itself (no adapter)
    training: dict[str, Any]  # options, mlx-lm adapter config, iterations reached
    name: str = ""

    def save(self, dest: str | Path) -> Path:
        """Write the self-contained artifact directory.

        Metadata and file copying do not require MLX; only ``LoRAModel`` does.
        The weights are checked for before anything is written, so an adapter
        path with no weights raises instead of silently saving a base model.
        """
        src = Path(self.adapter_path) if self.adapter_path is not None else None
        if src is not None and not (src / mlx.ADAPTER_WEIGHTS).is_file():
            raise FileNotFoundError(f"no {mlx.ADAPTER_WEIGHTS} in {src}")
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        if src is not None:
            for fname in (mlx.ADAPTER_WEIGHTS, mlx.ADAPTER_CONFIG):
                if (src / fname).exists():
                    shutil.copy2(src / fname, out / fname)
        (out / _META_FILE).write_text(
            json.dumps(
                {
                    "base_model": self.base_model,
                    "training": self.training,
                    "name": self.name,
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
            has_weights = (src / mlx.ADAPTER_WEIGHTS).exists()
            return cls(
                base_model=meta["base_model"],
                adapter_path=str(src) if has_weights else None,
                training=meta["training"],
                name=meta.get("name", ""),
            )
        return cls(base_model=str(src), adapter_path=None, training={})

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
        training: dict[str, Any] | None = None,
        name: str = "",
    ):
        mlx.require()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter_enabled = adapter_path is not None
        self.training = dict(training or {})
        self.name = name
        self._models: dict[bool, tuple[Any, Any]] = {}  # variant -> (model, tokenizer)

    def _variant(self, with_adapter: bool) -> tuple[Any, Any]:
        if with_adapter not in self._models:
            self._models[with_adapter] = mlx.load_model(
                self.base_model, self.adapter_path if with_adapter else None
            )
        return self._models[with_adapter]

    @property
    def tokenizer(self):
        """Travels with the model (kernel convention; shared by both variants)."""
        return self._variant(self.adapter_enabled)[1]

    def stream(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ):
        """The text as it is produced: the prompt echoed, then the completion in chunks.

        Temperature 0 is greedy and negative raises. The chunks are mlx-lm's, so
        they are arbitrary slices of text rather than words; concatenating them
        reproduces :meth:`generate`, which is exactly that concatenation. The
        first item is the prompt (echoed, per the kernel convention), so a caller
        can show it before the model has produced anything.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        model, tokenizer = self._variant(self.adapter_enabled)
        chunks = mlx.stream_text(
            model,
            tokenizer,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        first = next(chunks, None)
        if first is None:
            yield prompt
            return
        # mlx-lm returns the completion only, and does not say whether it starts
        # with a space, so the separator is decided once, against the first chunk.
        yield _echo(prompt, first)
        yield from chunks

    def generate(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> str:
        """The whole text: :meth:`stream` joined up, so there is one sampling path."""
        return "".join(self.stream(prompt, temperature, max_new_tokens, seed))

    def evaluate(self, text: str, max_tokens: int = 4000) -> dict:
        """Loss and perplexity over ``text``, in fixed 512-token windows.

        Returns ``{'loss', 'perplexity'}`` on the same convention as the Keras
        ``LanguageModel.evaluate``, for in-domain / out-of-domain tables. Only
        the first ``max_tokens`` ids are scored; fewer than two ids is undefined
        and returns NaN. ``adapter_enabled`` applies here too: turn it off to
        score the raw base model.
        """
        model, tokenizer = self._variant(self.adapter_enabled)
        return mlx.evaluate_text(model, tokenizer, text, max_tokens=max_tokens)

    def artifact(self) -> LoRAAdapter:
        """The model's adapter as a standalone artifact, ready to save."""
        return LoRAAdapter(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            training=self.training,
            name=self.name,
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> LoRAModel:
        """Load a model saved by :meth:`save`."""
        adapter = LoRAAdapter.load(path)
        return cls(
            adapter.base_model,
            adapter.adapter_path,
            adapter.training,
            adapter.name,
        )

    def save(self, dest: str | Path) -> Path:
        return self.artifact().save(dest)
