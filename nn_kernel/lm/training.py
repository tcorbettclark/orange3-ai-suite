"""Training entry point: corpus + options -> LanguageModel with embedded tokenizer.

Spec commitments (tiny-lm-suite-design.md §4.1):
- train/val split (~10%) is mandatory — train/val divergence IS the
  overfitting lesson on a tiny corpus (enforced here, not left to the widget);
- optimizer is Adam;
- live feedback: the kernel accepts a plain per-epoch hook (dict of epoch,
  loss, val_loss, perplexities) — the widget wraps it in a Keras callback;
  the kernel itself never sees Qt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from .builders import ARCHITECTURES, build_model, compile_model
from .models import LanguageModel
from .tokenizer import Tokenizer
from .windowing import make_windows

EpochStats = dict  # {epoch, loss, val_loss, perplexity, val_perplexity}


def train_language_model(
    texts: Iterable[str],
    tokenizer: Tokenizer,
    architecture: str,
    context_size: int = 16,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    lstm_units: int = 256,
    lstm_layers: int = 1,
    epochs: int = 15,
    batch_size: int = 256,
    validation_fraction: float = 0.1,
    learning_rate: float = 1e-3,
    seed: int = 0,
    on_epoch: Callable[[EpochStats], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    sample_prompt: str = "",
    sample_temperature: float | None = None,
    sample_max_new_tokens: int = 32,
    sample_seed: int = 0,
) -> LanguageModel:
    """Build windows, split by sentence, train, and return a LanguageModel.

    ``stop_requested``: checked after every epoch; when it returns True the
    Keras training loop stops cleanly (the widget's Stop button).

    When ``sample_temperature`` is not None, the hook's stats dict also
    carries ``"sample"`` — text generated from ``sample_prompt`` with the
    in-training model at each epoch (the checkpoint table: gibberish ->
    domain pastiche). The sampling loop lives here, in the kernel.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    windows = make_windows(tokenizer, texts, context_size)  # enforces sentence edges
    train, val = windows.split(validation_fraction, seed=seed)

    keras_model = compile_model(
        build_model(
            architecture,
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            seed=seed,
        ),
        learning_rate=learning_rate,
    )

    callbacks = []
    if on_epoch is not None or stop_requested is not None:
        import keras  # local: callback subclassing needs the real keras type

        class _EpochCallback(keras.callbacks.Callback):
            def __init__(self, emit, stop_requested):
                super().__init__()
                self._emit = emit
                self._stop_requested = stop_requested

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                if self._stop_requested is not None and self._stop_requested():
                    self.model.stop_training = True  # clean Keras early stop
                self._emit(epoch, logs or {})

        def _emit(epoch: int, logs: dict) -> None:
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            stats: dict = {
                "epoch": epoch + 1,
                "loss": float(loss) if loss is not None else float("nan"),
                "val_loss": float(val_loss) if val_loss is not None else float("nan"),
                "perplexity": math.exp(loss) if loss is not None else float("nan"),
                "val_perplexity": (
                    math.exp(val_loss) if val_loss is not None else float("nan")
                ),
            }
            if sample_temperature is not None:
                transient = LanguageModel(
                    architecture, tokenizer, context_size, keras_model, {}
                )
                stats["sample"] = transient.generate(
                    sample_prompt, sample_temperature, sample_max_new_tokens, sample_seed
                )
            on_epoch(stats)

        callbacks.append(_EpochCallback(_emit, stop_requested))

    # Bigram honours "context of 1": only the last token of a window enters
    # the model (builders.py documents this).
    if architecture == "bigram":
        def model_input(win):
            return win.contexts[:, -1:]
    else:
        def model_input(win):
            return win.contexts

    history = keras_fit(
        keras_model,
        model_input(train),
        train.targets,
        (model_input(val), val.targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    return LanguageModel(
        architecture=architecture,
        tokenizer=tokenizer,
        context_size=context_size,
        keras_model=keras_model,
        hyperparameters={
            "architecture": architecture,
            "context_size": context_size,
            "embedding_dim": embedding_dim,
            "hidden_units": hidden_units,
            "lstm_units": lstm_units,
            "lstm_layers": lstm_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_fraction": validation_fraction,
            "learning_rate": learning_rate,
            "seed": seed,
            "samples_enabled": sample_temperature is not None,
            "n_windows_total": len(windows),
            "n_train_windows": len(train),
            "n_val_windows": len(val),
            "final_loss": float(history["loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
        },
    )


def keras_fit(keras_model: Any, x, y, validation_data, *, epochs, batch_size, callbacks, verbose):
    """Thin Keras call kept in one place so the widget layer never reimplements it."""
    import keras  # local import, consistent with models.py

    return keras_model.fit(
        x,
        y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    ).history
