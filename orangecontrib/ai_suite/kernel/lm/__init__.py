"""Language-model kernel: tokenizers, pairs, architectures, and the model itself.

What this package exports is the use case, in the order it happens:

- **tokenize**: ``tokenizer_from_text`` (choose from ``tokenizers()``),
  returning a ``Tokenizer``.
- **choose an architecture**: ``Bigram``, ``MLP``, ``LSTM`` (enumerate them with
  ``architectures()``), each an ``Architecture``.
- **train and use**: ``LanguageModel(architecture, tokenizer).fit(text,
  TrainingOptions(...))``, then ``generate`` / ``evaluate``; the run's record is
  a ``TrainingResults``.
- **the layer below** ``fit``, for inspecting or reusing the data: ``make_pairs``
  returns a ``Pairs``.

Everything else lives in its module and is imported from there when a caller
needs it: the special tokens and their ids (``tokenizer``), the two concrete
tokenizers, ``tokenizer_from_dict`` (``tokenizer``), ``architecture_from_dict``
(``architecture``), ``EpochStats`` and the compile/callback helpers
(``training``).

``architecture`` imports Keras, and importing Keras imports its TensorFlow
backend, so importing this package costs a couple of seconds. That is paid once
per process and is why Orange takes a moment longer to start; it is not
deferrable, since every part of the model stack needs it.
"""

from .architecture import LSTM, MLP, Architecture, Bigram, architectures
from .language_model import LanguageModel
from .pairs import Pairs, make_pairs
from .tokenizer import Tokenizer, tokenizer_from_text, tokenizers
from .training import TrainingOptions, TrainingResults

__all__ = [
    "LSTM",
    "MLP",
    "Architecture",
    "Bigram",
    "LanguageModel",
    "Pairs",
    "Tokenizer",
    "TrainingOptions",
    "TrainingResults",
    "architectures",
    "make_pairs",
    "tokenizer_from_text",
    "tokenizers",
]
