"""The trained language-model object: an architecture with a tokenizer and weights.

:class:`LanguageModel` trains on a text, generates a unit, evaluates a text, and
saves its weights and tokenizer as one directory. Its ``generate`` signature is the
kernel's convention, shared with the MLX/LoRA models.
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

#: The model directory's own record. The name is shared with the LoRA artifact's, so the
#: record names the package that wrote it - and :meth:`LanguageModel.load` rejects one
#: that names anything else.
_META_FILE = "meta.json"
_CREATED_BY = "edu_ai_suite.lm"
_MODEL_FILE = "model.keras"


def _last_unit(stream: Iterable[int]) -> list[int]:
    """The final unit of a stream, without its EOS: an in-distribution prompt.

    An empty list when the stream carries no non-empty unit.
    """
    return next((unit for unit in reversed(stream_units(stream)) if unit), [])


def _sample(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Temperature sampling; 0 means greedy (argmax), and negatives are treated alike.

    A negative temperature only reaches this from a direct call: :meth:`LanguageModel.generate`
    rejects it.
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

    ``keras_model``, ``training`` and ``results`` are None until :meth:`fit` is
    called, and generating or scoring before that raises.
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
        """The Keras model holding the fitted weights; raises when there are none yet."""
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

        The split is made once, by unit, so the same units are held out in every epoch.
        ``self.keras_model`` is assigned before the first epoch, so an ``on_epoch`` hook
        may generate from the model as it stands.
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
        """Autoregressive sampling of one unit: its prompt echoed, then the completion.

        Generation stops at EOS, and ``max_new_tokens`` is a safety cap, so a model that
        never emits EOS returns a run-on. The prompt is trimmed to its final unit, and
        the returned text starts from that trimmed prompt rather than from ``prompt``.
        A sampled BOS is skipped and costs one step; a negative temperature raises.
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

    def stream(
        self,
        prompt: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ):
        """The text as it is produced - here, one chunk holding the whole of it.

        A caller concatenates the chunks and must not depend on where the boundaries fall.
        """
        yield self.generate(prompt, temperature, max_new_tokens, seed)

    def evaluate(self, text: str, batch_size: int = 256) -> dict[str, float]:
        """Loss and perplexity (e^loss) over the pairs of the given text.

        Perplexity is the loss re-scaled into "how many equally likely tokens the model
        behaves as if it were choosing between": 1 is perfect, and the vocabulary size
        is what a uniform guess scores.
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

        An unfitted model raises before anything is written, so no half-written
        directory is left behind.
        """
        keras_model = self._fitted
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "created_by": _CREATED_BY,
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
        """Load a model directory written by :meth:`save`.

        A ``meta.json`` written by any other package raises, rather than being read for
        whatever of its keys happen to match.
        """
        src = Path(path)
        meta = json.loads((src / _META_FILE).read_text())
        creator = meta.get("created_by")
        if creator != _CREATED_BY:
            raise ValueError(
                f"{src} was written by {creator!r}, not by {_CREATED_BY}: "
                "not a Keras language model"
            )
        return cls(
            architecture=architecture_from_dict(meta),
            tokenizer=tokenizer_from_dict(meta["tokenizer"]),
            keras_model=keras.models.load_model(src / _MODEL_FILE),
            training=(TrainingOptions(**meta["training"]) if meta.get("training") else None),
            results=(TrainingResults(**meta["results"]) if meta.get("results") else None),
        )
