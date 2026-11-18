"""Training settings, results, and the two Keras details that training owns.

An architecture describes a Keras graph and its own hyperparameters; training decides
the optimizer, the loss, how long to fit, when to stop and what counts as a
result. The loop itself lives on :class:`~.language_model.LanguageModel`, which
is the object that owns the weights; this module holds the nouns a run is
described and measured by, the optimizer/loss pair, and the epoch callback.

Sampling during training happens in the caller's ``on_epoch`` hook, not here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import keras

EpochStats = dict  # {epoch, loss, val_loss, perplexity, val_perplexity}, plus what a hook adds


@dataclass(frozen=True)
class TrainingOptions:
    """How to train: one run's settings, none of which describe the architecture.

    ``validation_fraction`` is the share of units held out, not of pairs: the
    division happens per unit, inside :meth:`Pairs.split`.
    """

    epochs: int = 15
    batch_size: int = 256
    validation_fraction: float = 0.1
    learning_rate: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")


@dataclass(frozen=True)
class TrainingResults:
    """What one run produced, beyond the weights: the counts and the final losses.

    Perplexity is ``e^loss``, derived from the loss rather than stored.
    """

    n_pairs_total: int
    n_train_pairs: int
    n_val_pairs: int
    final_loss: float
    final_val_loss: float

    @property
    def final_perplexity(self) -> float:
        return math.exp(self.final_loss)

    @property
    def final_val_perplexity(self) -> float:
        return math.exp(self.final_val_loss)


def compile_keras_model(keras_model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Attach optimizer and loss to a Keras model. This does not train it.

    The weights move for the first time in ``fit``; all this fixes is what that
    fit minimises. Sparse cross-entropy reads the model's raw logits, so softmax
    stays out of the Keras graph.
    """
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return keras_model


def epoch_callback(
    *,
    on_epoch: Callable[[EpochStats], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> keras.callbacks.Callback:
    """The two mid-run needs of a caller: stream stats, and stop between epochs.

    ``on_epoch`` receives one stats dict per epoch; a hook that wants to add a
    generated sample can call ``model.generate(...)`` itself, since the model being
    fitted is the caller's own object. ``stop_requested`` is polled after every
    epoch and ends the Keras loop cleanly rather than raising.
    """

    def emit(epoch: int, logs: dict) -> None:
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        stats: EpochStats = {
            "epoch": epoch + 1,
            "loss": float(loss) if loss is not None else float("nan"),
            "val_loss": float(val_loss) if val_loss is not None else float("nan"),
            "perplexity": math.exp(loss) if loss is not None else float("nan"),
            "val_perplexity": (math.exp(val_loss) if val_loss is not None else float("nan")),
        }
        on_epoch(stats)

    class _EpochCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            if stop_requested is not None and stop_requested():
                self.model.stop_training = True  # clean Keras early stop
            if on_epoch is not None:
                emit(epoch, logs or {})

    return _EpochCallback()
