"""Language-model kernel: tokenizers (with their statistics), windowing and
the Keras model ladder.

The Keras-dependent names (builders, training, ``LanguageModel``) resolve
lazily, so importing the tokenizer never pulls in TensorFlow. That keeps the
widget layer's optional ``[lm]`` extra genuinely optional.
"""

from .tokenizer import (
    BOS,
    BOS_ID,
    UNK,
    UNK_ID,
    CharTokenizer,
    Tokenizer,
    WordTokenizer,
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
    "CharTokenizer",
    "LanguageModel",
    "Tokenizer",
    "Windows",
    "WordTokenizer",
    "build_model",
    "compile_model",
    "make_windows",
    "split_sentences",
    "train_language_model",
]
