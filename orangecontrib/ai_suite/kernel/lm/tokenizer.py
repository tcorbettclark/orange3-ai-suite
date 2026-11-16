"""Tokenizers: a fixed vocabulary with corpus frequencies.

A tokenizer is a constituent part of a model, not a sibling of it: embedding
row *i* means token *i* only because this vocabulary says so. Trainers build
the tokenizer from the corpus and embed it in the emitted model object, so
vocabulary-mismatch bugs are unrepresentable. Instances are frozen: the
vocabulary travels with the weights.

Levels
------
Two concrete levels, each with its own segmentation and its own ``decode``
(they are different algorithms, not one algorithm with a flag):

- :class:`CharTokenizer` - every character is a token (spaces included - word
  boundaries matter at char level); ``decode`` concatenates, rendering the
  sentence separator as one space.
- :class:`WordTokenizer` - dependency-free regex tokens: words with internal
  apostrophes ("don't", Shakespeare's "o'er", both quote styles) plus every
  non-word non-space character as its own token. Whitespace is dropped, case is
  preserved, and ``decode`` re-inserts spaces except around punctuation.

Use :meth:`Tokenizer.from_corpus` to build either level.

Order
-----
Vocabulary order is an implementation detail and carries no meaning. Ids are
opaque: stable within one instance, and preserved verbatim by
``to_dict``/``from_dict`` so weights always index the vocabulary they were
trained with. Do not compare ids across tokenizers, and do not read anything
into the position of a token.

Anything that needs an order sorts explicitly. :meth:`Tokenizer.ranked_counts`
is the single definition of rank, ``(-count, token)``: frequency descending,
ties broken on token so that ranking - and therefore which hapax survives a
:meth:`Tokenizer.keep_top` cut - is deterministic and independent of corpus
order.

Special tokens (ids fixed across levels)
----------------------------------------
- id 0 = ``<bos>``: sentence-start marker, emitted by :meth:`Tokenizer.encode`
  before every sentence, and the context-window pad, so early positions with
  short context are trainable. Because the marker travels in the stream, the
  token domain is self-describing: ``[BOS, s1..sm, BOS, t1..tn]`` carries its
  own sentence structure and nothing downstream needs the text again. It is
  also indistinguishable from padding by design - both mean "nothing came
  before". Never a pruning casualty, and never a target: ``encode`` only puts
  it at sentence starts, and windowing never predicts index 0 of a sentence.
- id 1 = ``<unk>``: out-of-vocabulary input (unknown words at word level,
  unknown characters at char level, or tokens removed by pruning).

Both specials always survive pruning, and a pruning step that would remove
every real token raises instead of returning a degenerate vocabulary.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar

from ..text import split_sentences

BOS = "<bos>"
UNK = "<unk>"
BOS_ID = 0
UNK_ID = 1
SPECIAL_TOKENS = (BOS, UNK)

# In word-level decode these tokens close the previous token (no space before).
_ATTACH_PREV = set(".,;:!?…)]}%'’\"”»")
# And these open the next one (no space after). Only the unambiguous typographic
# quotes: straight quotes are symmetric and stay in _ATTACH_PREV, so 'hello'
# decodes as "' hello'" - a known limitation of reconstructing from tokens.
_ATTACH_NEXT = set("‘“«([{")


@dataclass(frozen=True)
class Tokenizer(ABC):
    """A frozen vocabulary with corpus frequencies.

    Abstract: build one with :meth:`from_corpus`, which dispatches on ``level``
    to :class:`CharTokenizer` or :class:`WordTokenizer`. ``total_token_count``
    is the number of tokens the corpus contained, including tokens that later
    pruning removed; it is what frequency *shares* are measured against, so
    shares stay comparable across pruning steps.
    """

    tokens: tuple[str, ...]  # id -> token, specials at ids 0 and 1
    counts: tuple[int, ...]  # corpus frequency per id (0 for specials)
    total_token_count: int | None = None  # corpus total; None = assume fully represented
    _token_to_id: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    level: ClassVar[str]

    def __post_init__(self) -> None:
        if len(self.counts) != len(self.tokens):
            raise ValueError("counts and tokens must have the same length")
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(f"vocabulary must start with the special tokens {SPECIAL_TOKENS}")
        ids = {t: i for i, t in enumerate(self.tokens)}
        if len(ids) != len(self.tokens):
            raise ValueError("duplicate tokens in vocabulary")
        object.__setattr__(self, "_token_to_id", ids)
        if self.total_token_count is None:
            object.__setattr__(self, "total_token_count", sum(self.counts))

    # ---- the level catalogue ---------------------------------------------

    @staticmethod
    def levels() -> tuple[tuple[str, str], ...]:
        """(level, one-line description) for each concrete level, in definition order.

        The description is the level class's own docstring summary, so the
        catalogue can never drift from the classes. Callers get both what to
        pass to :meth:`from_corpus` and a sentence to show a user.
        """
        return tuple(
            (sub.level, (inspect.getdoc(sub) or "").partition("\n")[0])
            for sub in Tokenizer.__subclasses__()
        )

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_corpus(cls, texts: Iterable[str], level: str) -> Tokenizer:
        """Count tokens over the documents and fix the vocabulary.

        Dispatches on ``level`` regardless of the class it is called on, so
        ``Tokenizer.from_corpus`` and ``CharTokenizer.from_corpus`` agree.

        Order is whatever the corpus presented first; by contract that means
        nothing. Prune afterwards with :meth:`keep_top` or :meth:`drop_rare`.
        """
        impl = cls._subclass_for(level)
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(impl.segment(text))
        if not counts:
            raise ValueError("corpus produced no tokens (empty or whitespace-only)")
        pairs = list(counts.items())
        return impl(
            tokens=tuple(SPECIAL_TOKENS) + tuple(t for t, _ in pairs),
            counts=tuple(0 for _ in SPECIAL_TOKENS) + tuple(c for _, c in pairs),
            total_token_count=sum(counts.values()),
        )

    @staticmethod
    def _subclass_for(level: str) -> type[Tokenizer]:
        """The concrete class for ``level``; every level sits directly under ``Tokenizer``."""
        for sub in Tokenizer.__subclasses__():
            if sub.level == level:
                return sub
        available = tuple(name for name, _ in Tokenizer.levels())
        raise ValueError(f"level must be one of {available}, got {level!r}")

    # ---- mapping ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def n_special(self) -> int:
        return len(SPECIAL_TOKENS)

    def token(self, index: int) -> str:
        return self.tokens[index]

    def index(self, token: str) -> int:
        return self._token_to_id[token]

    def is_special(self, index: int) -> bool:
        return 0 <= index < self.n_special

    def counted(self) -> list[tuple[str, int]]:
        """(token, corpus count) pairs for real tokens, in vocabulary order.

        Vocabulary order is meaningless by contract; use :meth:`ranked_counts`
        for anything user-visible.
        """
        n = self.n_special
        return list(zip(self.tokens[n:], self.counts[n:], strict=True))

    def ranked_counts(self) -> list[tuple[str, int]]:
        """(token, corpus count) pairs by ``(-count, token)``: the definition of rank."""
        return sorted(self.counted(), key=lambda tc: (-tc[1], tc[0]))

    @property
    def dropped_occurrences(self) -> int:
        """Corpus tokens no longer representable after pruning (they encode as ``<unk>``)."""
        return (self.total_token_count or 0) - sum(self.counts)

    # ---- pruning ---------------------------------------------------------

    def keep_top(self, n: int) -> Tokenizer:
        """A vocabulary of the ``n`` most frequent real tokens; specials always kept."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return self._rebuilt(self.ranked_counts()[:n])

    def drop_rare(
        self, *, min_count: int | None = None, min_share: float | None = None
    ) -> Tokenizer:
        """Drop tokens below a frequency floor, stated as a count or a corpus share.

        ``min_share`` is measured against ``total_token_count`` - the original
        corpus total - never against the survivors, so repeated pruning steps
        keep the same meaning.
        """
        if (min_count is None) == (min_share is None):
            raise ValueError("pass exactly one of min_count= or min_share=")
        if min_count is not None:
            if min_count < 1:
                raise ValueError("min_count must be >= 1")
            floor: float = min_count
        else:
            if not 0.0 < min_share <= 1.0:
                raise ValueError("min_share must be in (0, 1]")
            floor = min_share * (self.total_token_count or 0)
        return self._rebuilt([tc for tc in self.ranked_counts() if tc[1] >= floor])

    def _rebuilt(self, keep: list[tuple[str, int]]) -> Tokenizer:
        """A new tokenizer of the same level holding only ``keep``; total is preserved."""
        if not keep:
            raise ValueError(
                "criterion removes every real token "
                f"(vocabulary size {self.vocab_size}, corpus total {self.total_token_count})"
            )
        return type(self)(
            tokens=tuple(SPECIAL_TOKENS) + tuple(t for t, _ in keep),
            counts=tuple(0 for _ in SPECIAL_TOKENS) + tuple(c for _, c in keep),
            total_token_count=self.total_token_count,
        )

    # ---- encoding / decoding ----------------------------------------------
    #
    # encode() is the single text -> token transition, and it emits sentence
    # structure as data: a BOS before every sentence. After it the stream is
    # self-describing - no caller needs the text, and none needs a parallel
    # list of sentence lengths.

    def encode(self, text: str) -> list[int]:
        """Text -> a token-id stream with a BOS before every sentence.

        Sentence boundaries are detected on the raw text
        (:func:`~orangecontrib.ai_suite.kernel.text.split_sentences`), because
        tokenization discards the whitespace those rules depend on. A leading
        BOS appears even for a single sentence, so a stream always starts with
        one; ``decode`` drops BOS, so ``decode(encode(t))`` still round-trips.
        Out-of-vocabulary items map to ``<unk>``.
        """
        stream: list[int] = []
        for sentence in split_sentences(text):
            ids = [self._token_to_id.get(t, UNK_ID) for t in self.segment(sentence)]
            if ids:
                stream.append(BOS_ID)
                stream.extend(ids)
        return stream

    def encode_corpus(self, texts: Iterable[str]) -> list[int]:
        """Documents -> one marked stream, in corpus order.

        Concatenating *encoded* documents (not the texts) is what stops the
        last sentence of one document running into the first of the next:
        every sentence carries its own leading BOS.
        """
        return [i for text in texts for i in self.encode(text)]

    @classmethod
    @abstractmethod
    def segment(cls, text: str) -> list[str]:
        """Split text into token strings, before vocabulary lookup.

        A level-level operation, not an instance one: it is what
        :meth:`from_corpus` uses to count tokens before any instance exists.
        """

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text. ``<bos>`` markers are dropped; ``<unk>`` shows literally."""

    # ---- reporting ---------------------------------------------------------
    #
    # Read-only derivations from the counts: what the vocabulary holds and how
    # its frequencies are distributed. Pure queries - they never feed back into
    # training, generation or serialisation, so they can be called freely.

    def stats(self) -> dict:
        """Summary of the vocabulary: size, longest tokens, spread, hapax fraction, pruning."""
        real = self.counted()
        by_length = sorted(real, key=lambda tc: (-len(tc[0]), -tc[1], tc[0]))
        lengths = [len(t) for t, _ in real]
        n = len(real)
        return {
            "level": self.level,
            "vocab_size": self.vocab_size,
            "special_tokens": (self.token(BOS_ID), self.token(UNK_ID)),
            "longest_tokens": by_length[:10],
            "mean_token_length": (sum(lengths) / n) if n else 0.0,
            "single_occurrence_fraction": (sum(1 for _, c in real if c == 1) / n if n else 0.0),
            # Shares are corpus-relative, so pruning is visible rather than silent.
            "total_token_count": self.total_token_count,
            "dropped_occurrences": self.dropped_occurrences,
        }

    def frequency_table(self, n: int = 20) -> list[tuple[int, str, int]]:
        """Top-n tokens as (rank, token, count), 1-based ranks."""
        return [
            (rank, token, count)
            for rank, (token, count) in enumerate(self.ranked_counts()[:n], start=1)
        ]

    def zipf_summary(self, top: int = 100) -> dict:
        """Summary of the rank/count relation behind Zipf's law.

        For natural text rank x count is roughly constant. ``zipf_product_mean``
        is the mean of count*rank over the top ``top`` tokens;
        ``zipf_product_spread`` is its relative range. Shares are of the whole
        corpus, not of the retained vocabulary.
        """
        table = self.frequency_table(top)
        if not table:
            return {"n": 0}
        products = [count * rank for rank, _, count in table]
        total = self.total_token_count or 0
        mean_product = sum(products) / len(products)
        return {
            "n": len(table),
            "count_top1": table[0][2],
            "top1_share": table[0][2] / total if total else 0.0,
            "top10_share": sum(c for _, _, c in table[:10]) / total if total else 0.0,
            "zipf_product_mean": mean_product,
            "zipf_product_spread": (max(products) - min(products)) / mean_product,
        }

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-Python view for JSON serialisation next to the weights.

        Order is significant *here* even though it is undefined elsewhere: the
        saved list is what ids are read back from.
        """
        return {
            "level": self.level,
            "tokens": list(self.tokens),
            "counts": list(self.counts),
            "total_token_count": self.total_token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        return cls._subclass_for(data["level"])(
            tokens=tuple(data["tokens"]),
            counts=tuple(data["counts"]),
            # Absent in saves that predate pruning: fall back to the retained sum.
            total_token_count=data.get("total_token_count"),
        )


class CharTokenizer(Tokenizer):
    """Character level: every character is a token, spaces included."""

    level: ClassVar[str] = "char"

    @classmethod
    def segment(cls, text: str) -> list[str]:
        return list(text)

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text.

        Concatenation, so within a sentence this round-trips exactly. A
        non-leading BOS renders as one space: the sentence separator is not a
        token, so a newline or a run of spaces comes back as a single space.
        """
        pieces: list[str] = []
        for i in map(int, ids):
            if i == BOS_ID:
                if pieces:
                    pieces.append(" ")
            else:
                pieces.append(self.tokens[i])
        return "".join(pieces)


class WordTokenizer(Tokenizer):
    """Word level: regex words (internal apostrophes kept) plus lone punctuation."""

    level: ClassVar[str] = "word"

    # Words with internal apostrophes, then any single non-word non-space char.
    _PATTERN: ClassVar[re.Pattern] = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

    @classmethod
    def segment(cls, text: str) -> list[str]:
        return cls._PATTERN.findall(text)

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.tokens[i] for i in map(int, ids) if i != BOS_ID]
        if not tokens:
            return ""
        # Each token after the first either attaches to the previous one
        # (closing punctuation, or the previous token opens a quote) or starts
        # a new space-separated word. Pieces then join in one pass, so decoding
        # is linear regardless of whether the buffer is aliased.
        pieces = [tokens[0]] + [
            token if prev[-1] in _ATTACH_NEXT or token in _ATTACH_PREV else " " + token
            for prev, token in pairwise(tokens)
        ]
        return "".join(pieces)
