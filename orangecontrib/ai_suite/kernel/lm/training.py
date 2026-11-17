"""Training settings, results, and the two Keras details that training owns.

An architecture describes a graph and its own hyperparameters; training decides
the optimizer, the loss, how long to fit, when to stop and what counts as a
result. The loop itself lives on :class:`~.language_model.LanguageModel`, which
is the object that owns the weights; this module holds the nouns a run is
described and measured by, plus the per-epoch plumbing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import keras

EpochStats = dict  # {epoch, loss, val_loss, perplexity, val_perplexity[, sample]}


@dataclass(frozen=True)
class TrainingOptions:
    """How to train: one run's settings, none of which describe the architecture.

    ``sample_temperature`` is None unless the caller wants a generated sample
    after every epoch, which is what makes the checkpoint table readable: it is
    a snapshot of current behaviour rather than a number.
    """

    epochs: int = 15
    batch_size: int = 256
    validation_fraction: float = 0.1
    learning_rate: float = 1e-3
    seed: int = 0
    sample_prompt: str = ""
    sample_temperature: float | None = None
    sample_max_new_tokens: int = 32
    sample_seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")


@dataclass(frozen=True)
class TrainingResults:
    """What one run produced, beyond the weights: the counts and the final losses."""

    n_pairs_total: int
    n_train_pairs: int
    n_val_pairs: int
    final_loss: float
    final_val_loss: float


def compile_keras_model(keras_model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Attach optimizer and loss to a Keras model. This does not train it.

    The weights move for the first time in ``fit``; all this fixes is what that
    fit minimises. Sparse cross-entropy reads the model's raw logits, so softmax
    stays out of the graph.
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
    sample: Callable[[], str] | None = None,
) -> keras.callbacks.Callback:
    """The two mid-run needs of a caller: stream stats, and stop between epochs.

    ``on_epoch`` receives one stats dict per epoch. ``sample``, when given, is
    called once per epoch to add a generated sample under ``"sample"`` - it needs
    the model as it currently stands, so the caller supplies it rather than this
    module reaching for it. ``stop_requested`` is polled after every epoch and
    ends the Keras loop cleanly rather than raising.
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
        if sample is not None:
            stats["sample"] = sample()
        on_epoch(stats)

    class _EpochCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            if stop_requested is not None and stop_requested():
                self.model.stop_training = True  # clean Keras early stop
            if on_epoch is not None:
                emit(epoch, logs or {})

    return _EpochCallback()
