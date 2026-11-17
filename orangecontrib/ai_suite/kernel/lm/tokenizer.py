"""Tokenizers: a fixed vocabulary with corpus frequencies.

A tokenizer is a constituent part of a model.

Character and Word level tokenizers
-----------------------------------

There are two tokenizers, both subclasses of an abstract :class:`Tokenizer`:

- :class:`CharTokenizer` - every character is a token, including spaces.
- :class:`WordTokenizer` - every word is a token, identified with a regex.
  Words with internal apostrophes ("don't", Shakespeare's "o'er", both quote
  styles) plus every non-word non-space character as its own token. Whitespace
  is dropped, case is preserved, and rendering a unit re-inserts spaces except
  around punctuation.

Use :func:`tokenizer_from_text` to build either level.

Order
-----

Vocabulary order is an implementation detail and carries no meaning.

Ids are opaque, stable within one instance, and preserved verbatim by
``to_dict``/``from_dict`` so weights always index the vocabulary they were
trained with.

Do not compare ids across tokenizers, and do not read anything
into the position of a token.


Special tokens (ids fixed across levels)
----------------------------------------

- id 0 = ``<bos>``: the start-of-sequence marker, meaning "nothing came
  before". ``encode`` never emits it: the window builder prepends it as the
  start of a unit's context and uses it as the left pad, so early positions
  with short context stay trainable rather than being thrown away.
- id 1 = ``<unk>``: out-of-vocabulary input (unknown words at word level,
  unknown characters at char level, or tokens removed by pruning).
- id 2 = ``<eos>``: closes every unit, emitted by :meth:`Tokenizer.encode`. It
  is the only special that travels in the stream, and is the target so generation
  can learn to stop.

"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar

from ..text import split_sentences

BOS = "<bos>"
UNK = "<unk>"
EOS = "<eos>"
BOS_ID = 0
UNK_ID = 1
EOS_ID = 2
SPECIAL_TOKENS = (BOS, UNK, EOS)


@dataclass(frozen=True)
class Tokenizer(ABC):
    """A frozen vocabulary with corpus frequencies.

    Abstract: build one with :func:`tokenizer_from_text`, which dispatches on
    ``level`` to :class:`CharTokenizer` or :class:`WordTokenizer`.
    """

    tokens: tuple[str, ...]  # id -> token, specials first
    counts: tuple[int, ...]  # corpus frequency per id (0 for specials)
    total_token_count: int  # corpus tokens, including those pruning later removes
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
        return self.total_token_count - sum(self.counts)

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
            floor = min_share * self.total_token_count
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

    def encode(self, text: str) -> list[int]:
        """Text -> a token-id stream where every unit ends with EOS.

        This is the single text -> token transition. It marks the END of
        every unit with EOS. Starts (BOS) are not marked in the stream: the window
        builder prepends BOS as the start-of-context marker, so ends travel as
        data while starts are context construction.

        A unit is one sentence.

        Out-of-vocabulary items map to ``<unk>``

        :meth:`Tokenizer.decode` drops specials, so ``decode(encode(t))``
        round-trips up to the normalisation described there.
        """
        stream: list[int] = []
        for sentence in split_sentences(text):
            ids = [self._token_to_id.get(t, UNK_ID) for t in self.segment(sentence)]
            if ids:
                stream.extend(ids)
                stream.append(EOS_ID)
        return stream

    @classmethod
    @abstractmethod
    def segment(cls, text: str) -> list[str]:
        """Split text into token strings, before vocabulary lookup."""

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text: units rendered independently, joined by one space.

        Both specials are dropped, so ``<eos>`` closes a unit rather than
        appearing in the text. The separator between units is not a token, so a
        newline or a run of spaces comes back as a single space. ``<unk>``
        shows literally.
        """
        units: list[list[int]] = [[]]
        for i in map(int, ids):
            if i == EOS_ID:
                units.append([])
            elif i != BOS_ID:
                units[-1].append(i)
        return " ".join(text for text in map(self._decode_unit, units) if text)

    @abstractmethod
    def _decode_unit(self, ids: Sequence[int]) -> str:
        """Render one unit (no specials present) as text."""

    # ---- reporting ---------------------------------------------------------
    #
    # Read-only statistics from the vocabulary frequencies.

    def stats(self) -> dict:
        """Summary of the vocabulary: size, longest tokens, spread, hapax fraction, pruning."""
        real = self.counted()
        by_length = sorted(real, key=lambda tc: (-len(tc[0]), -tc[1], tc[0]))
        lengths = [len(t) for t, _ in real]
        n = len(real)
        return {
            "level": self.level,
            "vocab_size": self.vocab_size,
            "special_tokens": tuple(self.token(i) for i in range(self.n_special)),
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
        total = self.total_token_count
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

        Order is significant *here* even though it is undefined elsewhere.
        """
        return {
            "level": self.level,
            "tokens": list(self.tokens),
            "counts": list(self.counts),
            "total_token_count": self.total_token_count,
        }


class CharTokenizer(Tokenizer):
    """Character level: every character is a token, spaces included."""

    level: ClassVar[str] = "char"

    @classmethod
    def segment(cls, text: str) -> list[str]:
        return list(text)

    def _decode_unit(self, ids: Sequence[int]) -> str:
        return "".join(self.tokens[i] for i in ids)


class WordTokenizer(Tokenizer):
    """Word level: regex words (internal apostrophes kept) plus lone punctuation."""

    level: ClassVar[str] = "word"

    # Words with internal apostrophes, then any single non-word non-space char.
    _PATTERN: ClassVar[re.Pattern] = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

    # In word-level decode these tokens close the previous token (no space before).
    _ATTACH_PREV: ClassVar[frozenset[str]] = frozenset(".,;:!?…)]}%'’\"”»")
    # And these open the next one (no space after). Only the unambiguous typographic
    # quotes: straight quotes are symmetric and stay in _ATTACH_PREV, so 'hello'
    # decodes as "' hello'" - a known limitation of reconstructing from tokens.
    _ATTACH_NEXT: ClassVar[frozenset[str]] = frozenset("‘“«([{")

    @classmethod
    def segment(cls, text: str) -> list[str]:
        return cls._PATTERN.findall(text)

    def _decode_unit(self, ids: Sequence[int]) -> str:
        tokens = [self.tokens[i] for i in ids]
        if not tokens:
            return ""
        # Each token after the first either attaches to the previous one
        # (closing punctuation, or the previous token opens a quote) or starts
        # a new space-separated word. Pieces then join in one pass, so decoding
        # is linear regardless of whether the buffer is aliased.
        pieces = [tokens[0]] + [
            token if prev[-1] in self._ATTACH_NEXT or token in self._ATTACH_PREV else " " + token
            for prev, token in pairwise(tokens)
        ]
        return "".join(pieces)


# ---- construction -----------------------------------------------------------


def tokenizer_levels() -> tuple[tuple[str, str], ...]:
    """(level, one-line description) for each concrete level, in definition order.

    The level can be passed to :func:`tokenizer_from_text` to construct a
    concrete tokenizer; the description is the class's own docstring summary.
    """
    return tuple(
        (sub.level, (inspect.getdoc(sub) or "").partition("\n")[0])
        for sub in Tokenizer.__subclasses__()
    )


def _class_for(level: str) -> type[Tokenizer]:
    """The concrete class for ``level``.

    Only *direct* subclasses of Tokenizer are levels: Tokenizer.__subclasses__()
    does not see grandchildren, so a subclass of a level would be invisible here
    and in :func:`tokenizer_levels`.
    """
    for sub in Tokenizer.__subclasses__():
        if sub.level == level:
            return sub
    available = tuple(name for name, _ in tokenizer_levels())
    raise ValueError(f"level must be one of {available}, got {level!r}")


def tokenizer_from_text(text: str, level: str) -> Tokenizer:
    """Count tokens over the text and fix the vocabulary.

    ``level`` chooses the class, so this is the single entry point for both
    levels; :func:`tokenizer_levels` lists the names it accepts. A corpus of
    several documents arrives as one text: counts do not care where documents
    end, so nothing is lost by joining them first.
    """
    impl = _class_for(level)
    # Count what encode will actually tokenize: the sentences, not the raw text,
    # so a newline or the space after a terminator is never a token and
    # total_token_count stays the number of tokens the model can see.
    counts: Counter[str] = Counter(t for s in split_sentences(text) for t in impl.segment(s))
    if not counts:
        raise ValueError("corpus produced no tokens (empty or whitespace-only)")
    pairs = list(counts.items())
    return impl(
        tokens=tuple(SPECIAL_TOKENS) + tuple(t for t, _ in pairs),
        counts=tuple(0 for _ in SPECIAL_TOKENS) + tuple(c for _, c in pairs),
        total_token_count=sum(counts.values()),
    )


def tokenizer_from_dict(data: dict) -> Tokenizer:
    """Rebuild the tokenizer that :meth:`Tokenizer.to_dict` wrote."""
    return _class_for(data["level"])(
        tokens=tuple(data["tokens"]),
        counts=tuple(data["counts"]),
        total_token_count=data["total_token_count"],
    )
