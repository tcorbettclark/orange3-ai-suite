"""Language-model kernel: tokenizer, windowing, vocab inspector, Keras ladder.

Implemented: char/word tokenizers, BOS/padding, sentence-aware window
construction, vocab-inspector statistics, the Stage-1 architecture ladder
(bigram / MLP-over-window / LSTM) with Adam training, the autoregressive
sampling loop and the LanguageModel object whose tokenizer travels with it.
Pending: the MLX/LoRA side (nn_kernel.lora).
"""

from .tokenizer import BOS, UNK, BOS_ID, UNK_ID, Tokenizer
from .windowing import Windows, make_windows, split_sentences
from .vocab import (
    frequency_table,
    side_by_side,
    vocab_stats,
    zipf_summary,
)

# Keras-dependent names resolve lazily (PEP 562) so importing the Tokenizer or
# the Vocab Inspector logic NEVER pulls in TensorFlow — a hard requirement for
# the widget layer's dependency hygiene (arch doc: the kernel is import-light).
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
    "Tokenizer",
    "Windows",
    "make_windows",
    "split_sentences",
    "BOS",
    "UNK",
    "BOS_ID",
    "UNK_ID",
    "vocab_stats",
    "frequency_table",
    "zipf_summary",
    "side_by_side",
    "ARCHITECTURES",
    "build_model",
    "compile_model",
    "LanguageModel",
    "train_language_model",
]