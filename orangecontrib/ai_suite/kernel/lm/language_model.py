"""The trained language-model object.

A model is an instance of an architecture: the architecture is the recipe (a
Keras graph and its hyperparameters), the model is that recipe with a tokenizer and
weights. It is trained by :meth:`LanguageModel.fit`, and it generates,
evaluates, saves and loads. The ``generate`` interface is the kernel-wide
convention, shared with the MLX/LoRA models.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np

from .architecture import Architecture, architecture_from_dict
from .pairs import make_pairs
from .stats import EpochStats
from .tokenizer import BOS_ID, EOS_ID, Tokenizer, stream_units, tokenizer_from_dict
from .training import (
    TrainingOptions,
    TrainingResults,
    compile_keras_model,
    epoch_callback,
)

_META_FILE = "meta.json"
_MODEL_FILE = "model.keras"


def _last_unit(stream: Iterable[int]) -> list[int]:
    """The final unit of a stream, without its EOS: an in-distribution prompt.

    Prompting with anything before the last EOS would condition the model on a
    context it never saw in training, because contexts never cross a unit end.
    Unit structure is :func:`.tokenizer.stream_units`'s to derive.
    """
    return next((unit for unit in reversed(stream_units(stream)) if unit), [])


def _sample(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Temperature sampling; 0 means greedy (argmax), and negatives are treated alike.

    ``generate`` rejects a negative temperature, so only 0 reaches here from the
    public path.
    """
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
    """An architecture with a tokenizer and weights: fit it, generate from it.

    Implements the kernel's model convention, ``generate(prompt, temperature)``
    - the same interface the MLX/LoRA models expose. Nothing is trained until
    :meth:`fit` is called, and the width, the counts and the final losses all
    come from the architecture and from :class:`TrainingResults`, so there is no
    second copy of them here to drift.
    """

    architecture: Architecture
    tokenizer: Tokenizer
    keras_model: keras.Model | None = None
    training: TrainingOptions | None = None
    results: TrainingResults | None = None

    # ---- numbers that belong to the architecture ---------------------------

    @property
    def context_width(self) -> int:
        """How many ids of history the weights read: the architecture's own."""
        return self.architecture.context_width

    @property
    def _fitted(self) -> keras.Model:
        """The weights, or a clear error: there is nothing to generate from yet."""
        if self.keras_model is None:
            raise RuntimeError("the model is not fitted: call fit() first")
        return self.keras_model

    # ---- training ----------------------------------------------------------

    def fit(
        self,
        text: str,
        options: TrainingOptions | None = None,
        *,
        on_epoch: Callable[[EpochStats], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> LanguageModel:
        """Train on one text and return self, so the call can be chained.

        Pairs are built at the architecture's own context width and split by
        unit, so the same unit is held out for the whole run whatever the
        epoch. ``options`` carries everything about the run and nothing about
        the architecture. The Keras model is assigned
        before fitting begins, so an ``on_epoch`` hook can generate from the model
        as it currently stands - that is where a per-epoch sample belongs, not in
        the training settings.
        """
        options = options if options is not None else TrainingOptions()
        pairs = make_pairs(self.tokenizer.encode(text), self.context_width)
        train, val = pairs.split(options.validation_fraction, seed=options.seed)

        self.keras_model = compile_keras_model(
            self.architecture.build(vocab_size=self.tokenizer.vocab_size, seed=options.seed),
            options.learning_rate,
        )

        callbacks = []
        if on_epoch is not None or stop_requested is not None:
            callbacks.append(epoch_callback(on_epoch=on_epoch, stop_requested=stop_requested))

        history = self.keras_model.fit(
            train.contexts,
            train.targets,
            validation_data=(val.contexts, val.targets),
            epochs=options.epochs,
            batch_size=options.batch_size,
            callbacks=callbacks,
            verbose=0,
        ).history

        self.training = options
        self.results = TrainingResults(
            n_pairs_total=len(pairs),
            n_train_pairs=len(train),
            n_val_pairs=len(val),
            final_loss=float(history["loss"][-1]),
            final_val_loss=float(history["val_loss"][-1]),
        )
        return self

    # ---- generation and evaluation -----------------------------------------

    def generate(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> str:
        """Autoregressive sampling of one unit.

        Training, evaluation and generation share one unit, so this generates a
        sentence and stops when the model emits EOS. ``max_new_tokens`` is a
        safety cap: a model that never emits EOS returns a run-on. A sampled
        ``<bos>`` is not appended - it is never a target - so that step is
        spent. Asking for more than one unit would mean sampling from contexts
        that contain EOS - a distribution training never covered, because
        contexts never cross a unit end.

        The prompt is trimmed to its final unit for the same reason; contexts
        are the last ``context_width`` ids, left-padded with BOS.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        keras_model = self._fitted
        rng = np.random.default_rng(seed)
        prompt_ids = _last_unit(self.tokenizer.encode(prompt))
        # Left-pad the prompt's tail with BOS up to the context width; the pad
        # goes first, because a naive concatenation would exceed the width.
        tail = prompt_ids[-self.context_width :]
        context = [BOS_ID] * (self.context_width - len(tail)) + tail
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = keras_model(np.asarray([context], dtype=np.int32), training=False).numpy()[0]
            next_id = _sample(logits, temperature, rng)
            if next_id == EOS_ID:
                break
            if next_id == BOS_ID:
                continue  # start marker never appears in data; ignore a stray sample
            generated.append(next_id)
            context = context[1:] + [next_id]
        return self.tokenizer.decode(prompt_ids + generated)

    def evaluate(self, text: str, batch_size: int = 256) -> dict[str, float]:
        """Loss and perplexity (e^loss) over the pairs of the given text.

        Perplexity is the loss re-scaled into "how many equally likely tokens the
        model behaves as if it were choosing between": 1 would be perfect, and the
        vocabulary size is what a uniform guess scores. Run it on the training
        corpus and on unrelated text to get the in-domain / out-of-domain pair: a
        single-domain model looks good only on the domain it was trained on.
        """
        keras_model = self._fitted
        pairs = make_pairs(self.tokenizer.encode(text), self.context_width)
        loss = float(
            keras_model.evaluate(
                pairs.contexts,
                pairs.targets,
                batch_size=batch_size,
                verbose=0,
            )
        )
        return {"loss": loss, "perplexity": math.exp(loss)}

    def parameter_count(self) -> int:
        return int(self._fitted.count_params())

    # ---- persistence (vocab + weights as ONE unit) -------------------------

    def save(self, path: str | Path) -> Path:
        """Write weights, tokenizer and the run's record as one directory.

        The weights are checked for before anything is written, so an unfitted
        model raises without leaving a half-written directory behind.
        """
        keras_model = self._fitted
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "architecture_name": self.architecture.architecture_name(),
            "architecture_params": self.architecture.get_params(),
            "training": asdict(self.training) if self.training is not None else None,
            "results": asdict(self.results) if self.results is not None else None,
            "tokenizer": self.tokenizer.to_dict(),
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2))
        keras_model.save(out / _MODEL_FILE)
        return out

    @classmethod
    def load(cls, path: str | Path) -> LanguageModel:
        """Load a model directory saved by :meth:`save`."""
        src = Path(path)
        meta = json.loads((src / _META_FILE).read_text())
        return cls(
            architecture=architecture_from_dict(meta),
            tokenizer=tokenizer_from_dict(meta["tokenizer"]),
            keras_model=keras.models.load_model(src / _MODEL_FILE),
            training=(TrainingOptions(**meta["training"]) if meta.get("training") else None),
            results=(TrainingResults(**meta["results"]) if meta.get("results") else None),
        )
