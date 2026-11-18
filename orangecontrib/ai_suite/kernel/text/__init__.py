"""Plain text into training rows: cleaning, splitting and Q/A parsing.

Three steps, one for each kind of output:

- :mod:`._clean` - raw text to clean text: line endings, Gutenberg markers, page
  numbers, repeated headers, reflow of wrapped lines.
- :mod:`._split` - clean text to table rows: the whole file, one paragraph or one
  sentence per row. The model's units stay sentences; this only sets how many of
  them share a row.
- :mod:`._qa` - clean text to prompt/completion pairs, from fixed ``Q:``/``A:``
  blocks.

A caller picks a path through them: the text loader takes documents (clean,
then split) or Q/A (clean, then parse). Everything here works on strings and
returns frozen results, with no Orange or Qt import, so a caller only has to
deal with files and previews.
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
