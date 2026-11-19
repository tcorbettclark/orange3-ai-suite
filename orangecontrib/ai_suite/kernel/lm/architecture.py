"""The three from-scratch architectures: hyperparameters in, a Keras model out.

Bigram, MLP and LSTM share one input contract - a ``(context_width,)`` int32 context
of token ids - and one output contract - logits over the vocabulary, so softmax
lives in the loss and in the sampling loop. Each is a scikit-learn estimator that
stores hyperparameters verbatim, so ``get_params`` enumerates a model's settings;
the checks happen in :meth:`Architecture.build`. There is no transformer among
them.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import keras
from sklearn.base import BaseEstimator

from .tokenizer import SPECIAL_TOKENS


class Architecture(BaseEstimator, ABC):
    """One buildable architecture: hyperparameters in, a Keras model out."""

    @classmethod
    def architecture_name(cls) -> str:
        """The name this architecture is known by outside the kernel: ``"mlp"``."""
        return cls.__name__.lower()

    # How many ids of history the model reads: a hyperparameter where it is
    # tunable (MLP, LSTM), a constant where the architecture fixes it
    # (Bigram). The pair builder is told the same number, so data and model agree.
    context_width: int

    def build(self, *, vocab_size: int, seed: int = 0) -> keras.Model:
        """Build the Keras model: ``(n, context_width)`` ids -> ``(n, vocab_size)`` logits.

        The model reads :attr:`context_width` ids, the width the pairs must be built
        with. ``seed`` sets Keras's random seed, which is global to the process.
        """
        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise ValueError(
                f"vocab_size must be at least {len(SPECIAL_TOKENS) + 1}: "
                "the special tokens plus one real token"
            )
        if self.context_width < 1:
            raise ValueError("context_width must be >= 1")
        keras.utils.set_random_seed(seed)
        return self._keras_model(vocab_size=vocab_size)

    @abstractmethod
    def _keras_model(self, *, vocab_size: int) -> keras.Model:
        """The Keras model holding this architecture's layers, at the given vocabulary.

        Called from :meth:`build`, which has already seeded Keras and checked the
        arguments.
        """


class Bigram(Architecture):
    """Next token from the current token alone: the count table, learned.

    Takes no hyperparameters, and its context width is the constant 1, so its pairs
    must be built at width 1.
    """

    context_width = 1

    def _keras_model(self, *, vocab_size: int) -> keras.Model:
        inputs = keras.Input(shape=(self.context_width,), dtype="int32", name="token_ids")
        logits = keras.layers.Flatten(name="next_token_logits")(
            keras.layers.Embedding(vocab_size, vocab_size, name="bigram_table")(inputs)
        )
        return keras.Model(inputs, logits, name=type(self).__name__)


class MLP(Architecture):
    """The whole context concatenated into one hidden layer.

    Position is carried by concatenation order, so the model reads exactly
    :attr:`context_width` ids and no more.
    """

    def __init__(
        self,
        context_width: int = 16,
        embedding_dim: int = 32,
        hidden_units: int = 128,
    ):
        self.context_width = context_width
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units

    def _keras_model(self, *, vocab_size: int) -> keras.Model:
        inputs = keras.Input(shape=(self.context_width,), dtype="int32", name="token_ids")
        x = keras.layers.Embedding(vocab_size, self.embedding_dim, name="token_embeddings")(inputs)
        x = keras.layers.Flatten(name="concatenate_context")(x)
        x = keras.layers.Dense(self.hidden_units, activation="tanh", name="hidden")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)
        return keras.Model(inputs, logits, name=type(self).__name__)


class LSTM(Architecture):
    """Recurrence: position is carried by state rather than by concatenation."""

    def __init__(
        self,
        context_width: int = 16,
        embedding_dim: int = 32,
        units: int = 256,
        layers: int = 1,
    ):
        self.context_width = context_width
        self.embedding_dim = embedding_dim
        self.units = units
        self.layers = layers

    def _keras_model(self, *, vocab_size: int) -> keras.Model:
        if self.layers not in (1, 2):
            raise ValueError("layers must be 1 or 2")
        inputs = keras.Input(shape=(self.context_width,), dtype="int32", name="token_ids")
        x = keras.layers.Embedding(vocab_size, self.embedding_dim, name="token_embeddings")(inputs)
        for i in range(1, self.layers):
            x = keras.layers.LSTM(self.units, return_sequences=True, name=f"lstm_{i}")(x)
        x = keras.layers.LSTM(self.units, name="lstm_final")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)
        return keras.Model(inputs, logits, name=type(self).__name__)


def architectures() -> tuple[tuple[type[Architecture], str], ...]:
    """(architecture, one-line description) for each one, in definition order.

    The item is the class, and its description is the first line of that class's own
    docstring. Only direct subclasses of :class:`Architecture` are listed, so a
    subclass of a subclass is invisible here.
    """
    return tuple(
        (cls, (inspect.getdoc(cls) or "").partition("\n")[0])
        for cls in Architecture.__subclasses__()
    )


def _architecture(architecture_name: str) -> type[Architecture]:
    """The class an architecture name refers to."""
    for cls in Architecture.__subclasses__():
        if cls.architecture_name() == architecture_name:
            return cls
    available = tuple(cls.architecture_name() for cls in Architecture.__subclasses__())
    raise ValueError(f"architecture must be one of {available}, got {architecture_name!r}")


def architecture_from_dict(data: dict) -> Architecture:
    """Rebuild the architecture recorded by :meth:`LanguageModel.save`."""
    return _architecture(data["architecture_name"])(**data["architecture_params"])
