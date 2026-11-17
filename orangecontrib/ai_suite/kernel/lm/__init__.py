"""Language-model kernel: tokenizers (with their statistics), the pairs they feed,
the architectures, and the model that trains, generates and evaluates.

``architecture`` imports Keras, and importing Keras imports its TensorFlow backend,
so importing this package costs a couple of seconds. That is paid once per
process and is why Orange takes a moment longer to start; it is not deferrable,
since every part of the model stack needs it.
"""

from .architecture import (
    LSTM,
    MLP,
    Architecture,
    Bigram,
    architecture_from_dict,
    architectures,
)
from .language_model import LanguageModel
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
    tokenizers,
)
from .training import TrainingOptions, TrainingResults

__all__ = [
    "BOS",
    "BOS_ID",
    "EOS",
    "EOS_ID",
    "LSTM",
    "MLP",
    "UNK",
    "UNK_ID",
    "Architecture",
    "Bigram",
    "CharTokenizer",
    "LanguageModel",
    "Pairs",
    "Tokenizer",
    "TrainingOptions",
    "TrainingResults",
    "WordTokenizer",
    "architecture_from_dict",
    "architectures",
    "make_pairs",
    "tokenizer_from_dict",
    "tokenizer_from_text",
    "tokenizers",
]
