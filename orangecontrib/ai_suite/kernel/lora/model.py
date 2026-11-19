"""The base model with an optional LoRA adapter, each variant loaded on first use.

:class:`LoRAModel` follows the kernel's generation convention, shared with the Keras
``LanguageModel``: ``generate`` is ``stream`` joined up, and ``evaluate`` returns a loss
and a perplexity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import mlx_runtime
from .artifact import LoRAAdapter


def _echo(prompt: str, completion: str) -> str:
    """The prompt with the completion appended, one space between words.

    A space is inserted only when the prompt ends in a non-space and the completion
    starts with an alphanumeric character.
    """
    sep = (
        " "
        if prompt and completion and not prompt[-1].isspace() and completion[0].isalnum()
        else ""
    )
    return prompt + sep + completion


class LoRAModel:
    """A base model with an optional LoRA adapter; each variant loads on first use.

    ``adapter_enabled`` selects the raw base model or the adapter-fitted one, on the
    same weights; ``generate``, ``stream`` and ``evaluate`` follow the kernel's
    convention, shared with the Keras ``LanguageModel``.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        training: dict[str, Any] | None = None,
        name: str = "",
    ):
        mlx_runtime.require()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.adapter_enabled = adapter_path is not None
        self.training = dict(training or {})
        self.name = name
        self._models: dict[bool, tuple[Any, Any]] = {}  # variant -> (model, tokenizer)

    def _variant(self, with_adapter: bool) -> tuple[Any, Any]:
        if with_adapter not in self._models:
            self._models[with_adapter] = mlx_runtime.load_model(
                self.base_model, self.adapter_path if with_adapter else None
            )
        return self._models[with_adapter]

    @property
    def tokenizer(self):
        """The tokenizer of the loaded model, shared by both variants."""
        return self._variant(self.adapter_enabled)[1]

    def stream(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ):
        """The text as it is produced: the prompt echoed first, then chunks of completion.

        Temperature 0 is greedy and a negative raises. The chunks are mlx-lm's, so they
        are arbitrary slices of text rather than words: join them for the whole text and
        do not depend on where the boundaries fall.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        model, tokenizer = self._variant(self.adapter_enabled)
        chunks = mlx_runtime.stream_text(
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
        # mlx-lm returns the completion only, and does not report whether it starts
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
        """The whole text: :meth:`stream` joined up."""
        return "".join(self.stream(prompt, temperature, max_new_tokens, seed))

    def evaluate(self, text: str, max_tokens: int = 4000) -> dict:
        """Loss and perplexity over ``text``, in fixed 512-token windows.

        Returns ``{'loss', 'perplexity'}``, as the Keras ``LanguageModel.evaluate`` does.
        Only the first ``max_tokens`` ids are scored, and fewer than two ids returns NaN.
        ``adapter_enabled`` applies here too, so it can score the raw base model.
        """
        model, tokenizer = self._variant(self.adapter_enabled)
        return mlx_runtime.evaluate_text(model, tokenizer, text, max_tokens=max_tokens)

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
