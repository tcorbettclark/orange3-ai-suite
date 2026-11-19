"""Plain text as training rows: cleaning, splitting and Q/A parsing.

:mod:`._clean` produces clean text: line endings, Gutenberg markers, page numbers,
repeated headers and reflow of wrapped lines. :mod:`._split` produces table rows from
it, and :mod:`._qa` produces prompt/completion pairs from fixed ``Q:``/``A:`` blocks.

Nothing here imports Orange or Qt; the inputs and outputs are strings and frozen
results.
"""

from ._clean import CleanOptions, CleanResult, clean_text
from ._qa import QAPair, QAParseResult, parse_qa, reflow_qa_pair
from ._split import RowGranularity, split_sentences, split_text

__all__ = [
    "CleanOptions",
    "CleanResult",
    "QAPair",
    "QAParseResult",
    "RowGranularity",
    "clean_text",
    "parse_qa",
    "reflow_qa_pair",
    "split_sentences",
    "split_text",
]
