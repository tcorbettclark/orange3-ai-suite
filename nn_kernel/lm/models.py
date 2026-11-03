"""The model object: tokenizer embedded, generate(), save/load as one unit.

Spec commitments (tiny-lm-suite-design.md §4.2, §4.5):
- the tokenizer travels with the model — embedding row i means token i only
  because *this* tokenizer said so; MLX adapter models (later) share this
  interface, keyed on ``generate(prompt, temperature)``;
- Save/Load serialises vocabulary + weights as one unit (a directory: the
  Keras model file plus the tokenizer/meta JSON) — a separate tokenizer
  artifact is the vocab-mismatch bug by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .tokenizer import BOS_ID, Tokenizer
from .windowing import make_windows

_META_FILE = "meta.json"
_MODEL_FILE = "model.keras"


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

    Implements the kernel's model convention: ``generate(prompt, temperature)``
    — the same interface the MLX/LoRA models of Stages 2-3 will expose.
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
        """Autoregressive sampling: low temperature = repetitive, high = word salad.

        The prompt is encoded raw (no sentence splitting — it is a completion
        context); contexts are the last ``context_size`` ids, left-padded with
        <bos>. A sampled <bos> (which training never emits as a target) ends
        generation defensively.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        rng = np.random.default_rng(seed)
        prompt_ids = self.tokenizer.encode(prompt)
        # Left-pad the prompt's tail with <bos> up to the window size — note
        # the pad goes FIRST; a naive concatenation would exceed the window.
        tail = prompt_ids[-self.context_size :]
        context = [BOS_ID] * (self.context_size - len(tail)) + tail
        generated: list[int] = []
        for _ in range(max_new_tokens):
            window = context[-1:] if self.architecture == "bigram" else context
            logits = self.keras_model(
                np.asarray([window], dtype=np.int32), training=False
            ).numpy()[0]
            next_id = _sample(logits, temperature, rng)
            if next_id == BOS_ID:
                break
            generated.append(next_id)
            context = context[1:] + [next_id]
        return self.tokenizer.decode(prompt_ids + generated)

    def perplexity(
        self, texts: Iterable[str], batch_size: int = 256
    ) -> dict[str, float]:
        """Loss and perplexity (e^loss) over the windows of the given texts.

        This is the in-domain / out-of-domain number the evaluation spine plots
        (spec §6): the single-domain limitation as a chart.
        """
        windows = make_windows(self.tokenizer, texts, self.context_size)
        # bigram consumes only the last token of a window (builders.py)
        x = windows.contexts[:, -1:] if self.architecture == "bigram" else windows.contexts
        loss = float(
            self.keras_model.evaluate(
                x, windows.targets, batch_size=batch_size, verbose=0
            )
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
    def load(cls, path: str | Path) -> "LanguageModel":
        """Load a model directory saved by :meth:`save`.

        Uses keras.models.load_model at call time (import-light module).
        """
        import keras  # local import: only needed when materialising weights

        src = Path(path)
        meta = json.loads((src / _META_FILE).read_text())
        tokenizer = Tokenizer.from_dict(meta["tokenizer"])
        return cls(
            architecture=meta["architecture"],
            tokenizer=tokenizer,
            context_size=meta["context_size"],
            keras_model=keras.models.load_model(src / _MODEL_FILE),
            hyperparameters=meta["hyperparameters"],
        )