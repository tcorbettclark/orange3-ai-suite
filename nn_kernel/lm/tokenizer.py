"""Tokenizer: a fixed vocabulary with corpus frequencies.

The tokenizer is a constituent part of a model, not a sibling of it: embedding
row *i* means token *i* only because this vocabulary says so. Trainers build
the tokenizer from the corpus and embed it in the emitted model object, so
vocabulary-mismatch bugs are unrepresentable.

Levels
------
- ``"char"``: every character is a token (spaces included - word boundaries
  matter at char level).
- ``"word"``: dependency-free regex tokens - words with internal apostrophes
  ("don't", Shakespeare's "o'er", both quote styles) plus every non-word
  non-space character as its own token. Whitespace is dropped. Case is
  preserved (no silent normalisation).

Special tokens (ids fixed across levels)
----------------------------------------
- id 0 = ``<bos>``: sentence-start marker, and the context-window pad, so
  early positions with short context are trainable. Windowing places it; the
  tokenizer only owns the mapping.
- id 1 = ``<unk>``: out-of-vocabulary input (unknown words at word level,
  unknown characters at char level, or tokens removed by ``max_vocab``).

Vocabulary order is corpus-order-independent: sort by (-count, token).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

BOS = "<bos>"
UNK = "<unk>"
BOS_ID = 0
UNK_ID = 1
SPECIAL_TOKENS = (BOS, UNK)

LEVELS = ("char", "word")

# Words with internal apostrophes, then any single non-word non-space char.
_WORD_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

# In word-level decode these tokens attach to the previous word (no space).
_PUNCT_NO_SPACE = set(".,;:!?…)]}%'’\"")


@dataclass
class Tokenizer:
    """A fixed vocabulary with corpus frequencies.

    Created via :meth:`from_corpus`. Treat instances as read-only: the
    vocabulary travels with the model.
    """

    level: str
    tokens: tuple[str, ...]  # id -> token, specials at ids 0 and 1
    counts: tuple[int, ...]  # corpus frequency per id (0 for specials)
    _token_to_id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._token_to_id = {t: i for i, t in enumerate(self.tokens)}
        if len(self._token_to_id) != len(self.tokens):
            raise ValueError("duplicate tokens in vocabulary")

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        texts: Iterable[str],
        level: str,
        max_vocab: int | None = None,
    ) -> Tokenizer:
        """Build the vocabulary from documents.

        ``max_vocab`` (if set) is the total vocabulary size *including* the
        two special tokens; least-frequent tokens are dropped (they then
        encode as ``<unk>``).
        """
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_segment_tokens(text, level))
        if not counts:
            raise ValueError("corpus produced no tokens (empty or whitespace-only)")
        if max_vocab is not None and max_vocab < len(SPECIAL_TOKENS):
            raise ValueError(f"max_vocab must be >= {len(SPECIAL_TOKENS)}")
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_vocab is not None:
            ordered = ordered[: max_vocab - len(SPECIAL_TOKENS)]
        return cls(
            level=level,
            tokens=tuple(SPECIAL_TOKENS) + tuple(t for t, _ in ordered),
            counts=tuple(0 for _ in SPECIAL_TOKENS) + tuple(c for _, c in ordered),
        )

    # ---- mapping ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def token(self, index: int) -> str:
        return self.tokens[index]

    def index(self, token: str) -> int:
        return self._token_to_id[token]

    def is_special(self, index: int) -> bool:
        return index in (BOS_ID, UNK_ID)

    # ---- encoding / decoding ----------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Text -> token ids; out-of-vocabulary items map to ``<unk>``."""
        return [self._token_to_id.get(t, UNK_ID) for t in _segment_tokens(text, self.level)]

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text. ``<bos>`` markers are dropped; ``<unk>`` shows literally.

        Word level: punctuation attaches to the previous word (no space).
        Char level: plain concatenation (round-trips exactly).
        """
        pieces: list[str] = []
        for i in ids:
            i = int(i)
            if i == BOS_ID:
                continue
            token = self.tokens[i]
            if self.level == "word" and pieces and token in _PUNCT_NO_SPACE:
                pieces[-1] += token
            elif self.level == "char" or not pieces:
                pieces.append(token)
            else:
                pieces.append(" " + token)
        return "".join(pieces)

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-Python view for JSON serialisation next to the weights."""
        return {
            "level": self.level,
            "tokens": list(self.tokens),
            "counts": list(self.counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        return cls(
            level=data["level"],
            tokens=tuple(data["tokens"]),
            counts=tuple(data["counts"]),
        )


def _segment_tokens(text: str, level: str) -> list[str]:
    if level == "char":
        return list(text)
    return _WORD_TOKEN_RE.findall(text)
