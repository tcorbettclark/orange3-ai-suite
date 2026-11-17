"""Language-model kernel: tokenizers (with their statistics), the pairs they
feed, and the from-scratch Keras models with their training entry point.

``builders`` imports Keras, and importing Keras imports its TensorFlow backend,
so importing this package costs a couple of seconds. That is paid once per
process and is why Orange takes a moment longer to start; it is not deferrable,
since every part of the model stack needs it.
"""

from .builders import ARCHITECTURES, LSTM, MLP, Bigram, compile_model
from .models import LanguageModel
from .pairs import Pairs, make_pairs
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

__all__ = [
    "ARCHITECTURES",
    "BOS",
    "BOS_ID",
    "EOS",
    "EOS_ID",
    "LSTM",
    "MLP",
    "UNK",
    "UNK_ID",
    "Bigram",
    "CharTokenizer",
    "LanguageModel",
    "Pairs",
    "Tokenizer",
    "WordTokenizer",
    "compile_model",
    "make_pairs",
    "tokenizer_from_dict",
    "tokenizer_from_text",
    "tokenizer_levels",
    "train_language_model",
]
