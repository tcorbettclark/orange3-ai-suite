"""Keras model builders for the architecture ladder.

The ladder is bigram -> MLP-over-window -> LSTM. All architectures take the
same input contract, a ``(context_size,)`` int32 token-id window, and emit
logits over the vocabulary; softmax lives in the loss and in the sampling loop.
Only stock Keras layers are used: Embedding, Dense and LSTM.

No transformer is offered here. At classroom scale an LSTM is at least as
accurate and far cheaper to train on CPU; the transformer lesson is delivered
by adapting a real pretrained model with LoRA instead.
"""

from __future__ import annotations

import keras

ARCHITECTURES = ("bigram", "mlp_window", "lstm")


def build_model(
    architecture: str,
    *,
    vocab_size: int,
    context_size: int,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    lstm_units: int = 256,
    lstm_layers: int = 1,
    seed: int = 0,
) -> keras.Model:
    """Build one architecture of the ladder; all emit logits (vocab_size,)."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    if vocab_size < 3:
        raise ValueError("vocab_size must include at least <bos> and <unk> plus one token")
    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    if architecture == "lstm" and lstm_layers not in (1, 2):
        raise ValueError("lstm_layers must be 1 or 2")
    keras.utils.set_random_seed(seed)

    if architecture == "bigram":
        # Context of 1: the model consumes only the last token of a window
        # (LanguageModel/training slice `[:, -1:]`). Row i of the embedding
        # matrix is the next-token logit vector given token i - the bigram
        # count table, learned by gradient descent. Flattened to (vocab,).
        inputs = keras.Input(shape=(1,), dtype="int32", name="token_ids")
        logits = keras.layers.Flatten(name="next_token_logits")(
            keras.layers.Embedding(vocab_size, vocab_size, name="bigram_table")(inputs)
        )
        return keras.Model(inputs, logits, name=architecture)

    inputs = keras.Input(shape=(context_size,), dtype="int32", name="token_ids")
    if architecture == "mlp_window":
        x = keras.layers.Embedding(vocab_size, embedding_dim, name="token_embeddings")(inputs)
        x = keras.layers.Flatten(name="concatenate_window")(x)
        x = keras.layers.Dense(hidden_units, activation="tanh", name="hidden")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)
    else:  # lstm
        x = keras.layers.Embedding(vocab_size, embedding_dim, name="token_embeddings")(inputs)
        for i in range(1, lstm_layers):
            x = keras.layers.LSTM(lstm_units, return_sequences=True, name=f"lstm_{i}")(x)
        x = keras.layers.LSTM(lstm_units, name="lstm_final")(x)
        logits = keras.layers.Dense(vocab_size, name="logits")(x)

    return keras.Model(inputs, logits, name=architecture)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile with Adam and sparse cross-entropy over raw logits."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model
