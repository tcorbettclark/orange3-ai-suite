"""Language-model kernel: tokenizer, windowing, (builders coming).

Phase 1 (implemented): char/word tokenizers, BOS/padding, sentence-aware
window construction, vocab-inspector statistics. Pure Python/NumPy.
Phase 2 (pending): bigram/MLP-window/LSTM Keras builders and the sampling
loop, per spec/tiny-lm-suite-design.md §3-4.
"""

from .tokenizer import BOS, UNK, BOS_ID, UNK_ID, Tokenizer
from .windowing import Windows, make_windows, split_sentences
from .vocab import (
    frequency_table,
    side_by_side,
    vocab_stats,
    zipf_summary,
)

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
]