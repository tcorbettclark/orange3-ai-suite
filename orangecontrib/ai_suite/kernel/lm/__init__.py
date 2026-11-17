"""Language-model kernel: tokenizers (with their statistics), windowing, and the
from-scratch Keras models with their training entry point.

``builders`` imports Keras, and importing Keras imports its TensorFlow backend,
so importing this package costs a couple of seconds. That is paid once per
process and is why Orange takes a moment longer to start; it is not deferrable,
since every part of the model stack needs it.
"""

from .builders import ARCHITECTURES, LSTM, Bigram, MLPWindow, compile_model
from .models import LanguageModel
from .tokenizer import (
    BOS,
    BOS_ID,
    EOS,
    EOS_ID,
    UNK,
    UNK_ID,
    CharTokenizer,
    Tokenizer,
    WordTokenizer,
    tokenizer_from_dict,
    tokenizer_from_text,
    tokenizer_levels,
)
from .training import train_language_model
from .windowing import Windows, make_windows

__all__ = [
    "ARCHITECTURES",
    "BOS",
    "BOS_ID",
    "EOS",
    "EOS_ID",
    "LSTM",
    "UNK",
    "UNK_ID",
    "Bigram",
    "CharTokenizer",
    "LanguageModel",
    "MLPWindow",
    "Tokenizer",
    "Windows",
    "WordTokenizer",
    "compile_model",
    "make_windows",
    "tokenizer_from_dict",
    "tokenizer_from_text",
    "tokenizer_levels",
    "train_language_model",
]
