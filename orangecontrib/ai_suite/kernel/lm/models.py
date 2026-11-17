"""The trained language-model object.

The model owns its tokenizer, exposes ``generate()`` and ``perplexity()``, and
saves and loads vocabulary and weights as one unit. The ``generate`` interface
is the kernel-wide convention, shared with the MLX/LoRA models.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tokenizer import BOS_ID, EOS_ID, Tokenizer, tokenizer_from_dict
from .windowing import make_windows

_META_FILE = "meta.json"
_MODEL_FILE = "model.keras"


def _last_unit(stream: Iterable[int]) -> list[int]:
    """The final unit of a stream, without its EOS: an in-distribution prompt.

    Prompting with anything before the last EOS would condition the model on a
    context it never saw in training, because contexts never cross a unit end.
    """
    units: list[list[int]] = [[]]
    for i in stream:
        if i == EOS_ID:
            units.append([])
        elif i != BOS_ID:
            units[-1].append(i)
    return next((unit for unit in reversed(units) if unit), [])


def _sample(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Temperature sampling; temperature <= 0 means greedy (argmax)."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


@dataclass
class LanguageModel:
    """A trained Keras LM: architecture + tokenizer + weights + hyperparameters.

    Implements the kernel's model convention, ``generate(prompt, temperature)``
    - the same interface the MLX/LoRA models expose.
    """

    architecture: str
    tokenizer: Tokenizer
    context_size: int
    keras_model: Any  # keras.Model, typed loosely to keep the kernel import-light
    hyperparameters: dict

    def generate(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> str:
        """Autoregressive sampling of one unit: low temperature = repetitive, high = salad.

        Training, evaluation and generation share one unit, so this generates a
        sentence and stops when the model emits EOS. ``max_new_tokens`` is only
        a safety cap: an undertrained model that never emits EOS still returns a
        run-on, which is the honest picture. Asking for more than one unit would
        mean sampling from contexts that contain EOS - a distribution training
        never covered, because contexts never cross a unit end.

        The prompt is trimmed to its final unit for the same reason; contexts
        are the last ``context_size`` ids, left-padded with BOS.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        rng = np.random.default_rng(seed)
        prompt_ids = _last_unit(self.tokenizer.encode(prompt))
        # Left-pad the prompt's tail with BOS up to the window size; the pad
        # goes first, because a naive concatenation would exceed the window.
        tail = prompt_ids[-self.context_size :]
        context = [BOS_ID] * (self.context_size - len(tail)) + tail
        generated: list[int] = []
        for _ in range(max_new_tokens):
            window = context[-1:] if self.architecture == "bigram" else context
            logits = self.keras_model(np.asarray([window], dtype=np.int32), training=False).numpy()[
                0
            ]
            next_id = _sample(logits, temperature, rng)
            if next_id == EOS_ID:
                break
            if next_id == BOS_ID:
                continue  # start marker never appears in data; ignore a stray sample
            generated.append(next_id)
            context = context[1:] + [next_id]
        return self.tokenizer.decode(prompt_ids + generated)

    def perplexity(self, text: str, batch_size: int = 256) -> dict[str, float]:
        """Loss and perplexity (e^loss) over the windows of the given text.

        Run it on the training corpus and on unrelated text to get the
        in-domain / out-of-domain pair: a single-domain model looks good only on
        the domain it was trained on.
        """
        windows = make_windows(self.tokenizer.encode(text), self.context_size)
        # bigram consumes only the last token of a window (builders.py)
        x = windows.contexts[:, -1:] if self.architecture == "bigram" else windows.contexts
        loss = float(
            self.keras_model.evaluate(x, windows.targets, batch_size=batch_size, verbose=0)
        )
        return {"loss": loss, "perplexity": math.exp(loss)}

    def parameter_count(self) -> int:
        return int(self.keras_model.count_params())

    # ---- persistence (vocab + weights as ONE unit) -------------------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "architecture": self.architecture,
            "context_size": self.context_size,
            "hyperparameters": self.hyperparameters,
            "tokenizer": self.tokenizer.to_dict(),
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2))
        self.keras_model.save(out / _MODEL_FILE)
        return out

    @classmethod
    def load(cls, path: str | Path) -> LanguageModel:
        """Load a model directory saved by :meth:`save`."""
        import keras  # local import: only needed when materialising weights

        src = Path(path)
        meta = json.loads((src / _META_FILE).read_text())
        tokenizer = tokenizer_from_dict(meta["tokenizer"])
        return cls(
            architecture=meta["architecture"],
            tokenizer=tokenizer,
            context_size=meta["context_size"],
            keras_model=keras.models.load_model(src / _MODEL_FILE),
            hyperparameters=meta["hyperparameters"],
        )
