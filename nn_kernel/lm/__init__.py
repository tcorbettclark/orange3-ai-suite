"""Language-model kernel: tokenizers, windowing, vocab statistics and the
Keras model ladder.

The Keras-dependent names (builders, training, ``LanguageModel``) resolve
lazily, so importing the tokenizer or the vocab-inspector logic never pulls in
TensorFlow. That keeps the widget layer's optional ``[lm]`` extra genuinely
optional.
"""

from .tokenizer import BOS, BOS_ID, UNK, UNK_ID, Tokenizer
from .vocab import (
    frequency_table,
    side_by_side,
    vocab_stats,
    zipf_summary,
)
from .windowing import Windows, make_windows, split_sentences

# Keras-dependent names resolve lazily (PEP 562).
_KERAS_NAMES = {
    "ARCHITECTURES": "builders",
    "build_model": "builders",
    "compile_model": "builders",
    "LanguageModel": "models",
    "train_language_model": "training",
}


def __getattr__(name: str):
    module = _KERAS_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


__all__ = [
    "ARCHITECTURES",
    "BOS",
    "BOS_ID",
    "UNK",
    "UNK_ID",
    "LanguageModel",
    "Tokenizer",
    "Windows",
    "build_model",
    "compile_model",
    "frequency_table",
    "make_windows",
    "side_by_side",
    "split_sentences",
    "train_language_model",
    "vocab_stats",
    "zipf_summary",
]
