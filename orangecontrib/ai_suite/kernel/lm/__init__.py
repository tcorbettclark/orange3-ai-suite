"""Language-model kernel: tokenizers (with their statistics), windowing, and the
from-scratch Keras models with their training entry point.

Nothing here imports TensorFlow at module import: Keras is imported inside the
functions that build or run a graph, so the widget layer can import the kernel -
and describe an architecture, its parameters and a vocabulary - without the
optional ``[lm]`` extra. Only running training needs it.
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
