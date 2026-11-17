"""Keras model builders for the three from-scratch architectures.

Bigram, MLP-over-window and LSTM are separate models that happen to share an
input contract - a ``(context_size,)`` int32 token-id window - and an output
contract - logits over the vocabulary, so softmax lives in the loss and in the
sampling loop. They are offered side by side for comparison, not as one thing
that grows: each declares its own hyperparameters, so no model can be handed a
parameter it would ignore.

Each is a scikit-learn estimator: the constructor stores hyperparameters
verbatim and does nothing else, every check happens in :meth:`Architecture.build`,
and ``get_params`` / ``set_params`` / ``clone`` come from ``BaseEstimator``. That
lets a caller enumerate a model's parameters instead of guessing a superset -
Orange's controls and parameter sweeps are driven from exactly this.

No transformer is offered here. At classroom scale an LSTM is at least as
accurate and far cheaper to train on CPU; the transformer lesson is delivered
by adapting a real pretrained model with LoRA instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import keras
import numpy as np
from sklearn.base import BaseEstimator

from .tokenizer import SPECIAL_TOKENS


class Architecture(BaseEstimator, ABC):
    """One buildable model: hyperparameters in, a Keras graph out."""

    name: ClassVar[str]  # registry key, and what saved metadata records
    label: ClassVar[str]  # human-readable, for a dropdown

    def take(self, contexts: np.ndarray) -> np.ndarray:
        """The ids this model consumes from an ``(n, context_size)`` window batch.

        Everything that needs to feed a window - training, perplexity,
        generation - goes through here, so "this model only looks at the last
        token" is stated once, by the model that has that property.
        """
        return contexts

    def build(self, *, vocab_size: int, context_size: int, seed: int = 0) -> keras.Model:
        """Build the graph: ``(n, context_size)`` ids -> ``(n, vocab_size)`` logits.

        ``context_size`` is the width of the window the data provides; a model
        whose :meth:`take` narrows the window shapes its input to what it takes.
        Seeding lives here because it is a global Keras side effect, and this is
        the one place that builds a graph.
        """
        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise ValueError(
                f"vocab_size must be at least {len(SPECIAL_TOKENS) + 1}: "
                "the special tokens plus one real token"
            )
        if context_size < 1:
            raise ValueError("context_size must be >= 1")
        keras.utils.set_random_seed(seed)
        return self._graph(vocab_size=vocab_size, context_size=context_size)

    @abstractmethod
    def _graph(self, *, vocab_size: int, context_size: int) -> keras.Model:
        """The layers; seeding and input-independent checks live in :meth:`build`."""


class Bigram(Architecture):
    """Next token from the current token alone: the count table, learned.

    Takes no hyperparameters, because it has none: row *i* of the
    ``(vocab_size, vocab_size)`` embedding is the next-token logit vector given
    token *i*. That is the bigram count table that ``ranked_counts`` would
    describe, fitted by gradient descent instead of counting - which is the
    point of starting the comparison here.
    """

    name = "bigram"
    label = "Bigram"

    def take(self, contexts: np.ndarray) -> np.ndarray:
        # A bigram cannot use more than the token it conditions on.
        return contexts[:, -1:]

    def _graph(self, *, vocab_size: int, context_size: int) -> keras.Model:
        inputs = keras.Input(shape=(1,), dtype="int32", name="token_ids")
        logits = keras.layers.Flatten(name="next_token_logits")(
            keras.layers.Embedding(vocab_size, vocab_size, name="bigram_table")(inputs)
        )
        return keras.Model(inputs, logits, name=self.name)


class MLPWindow(Architecture):
    """A fixed, visible memory: the whole window concatenated into one hidden layer.

    Position is carried by concatenation order, so the model can only ever look
    ``context_size`` tokens back - the contrast with the LSTM, which carries
    position in its state.
    """

    name = "mlp_window"
    label = "MLP over window"

    def __init__(self, embedding_dim: int = 32, hidden_units: int = 128):
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units

    def _graph(self, *, vocab_size: int, context_size: int) -> keras.Model:
        inputs = keras.Input(shape=(context_size,), dtype="int32", name="token_ids")
        x = keras.layers.Embedding(vocab_size, self.embedding_dim, name="token_embeddings")(inputs)
        x = keras.layers.Flatten(name="concatenate_window")(x)
        x = keras.layers.Dense(self.hidden_units, activation="tanh", name="hidden")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)
        return keras.Model(inputs, logits, name=self.name)


class LSTM(Architecture):
    """Recurrence: position is carried by state rather than by concatenation."""

    name = "lstm"
    label = "LSTM"

    def __init__(self, embedding_dim: int = 32, units: int = 256, layers: int = 1):
        self.embedding_dim = embedding_dim
        self.units = units
        self.layers = layers

    def _graph(self, *, vocab_size: int, context_size: int) -> keras.Model:
        if self.layers not in (1, 2):
            raise ValueError("layers must be 1 or 2")
        inputs = keras.Input(shape=(context_size,), dtype="int32", name="token_ids")
        x = keras.layers.Embedding(vocab_size, self.embedding_dim, name="token_embeddings")(inputs)
        for i in range(1, self.layers):
            x = keras.layers.LSTM(self.units, return_sequences=True, name=f"lstm_{i}")(x)
        x = keras.layers.LSTM(self.units, name="lstm_final")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)
        return keras.Model(inputs, logits, name=self.name)


ARCHITECTURES: dict[str, type[Architecture]] = {cls.name: cls for cls in (Bigram, MLPWindow, LSTM)}


def architecture_from_metadata(meta: dict) -> Architecture:
    """Rebuild the architecture recorded by :meth:`LanguageModel.save`."""
    cls = ARCHITECTURES[meta["name"]]
    return cls(**meta["params"])


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile with Adam and sparse cross-entropy over raw logits."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model
