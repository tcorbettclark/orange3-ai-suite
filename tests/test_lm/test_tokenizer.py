"""Tokenizer edge cases: special ids, the catalogue, encode/decode round trips, pruning
(BOS/padding, unknowns, sentence boundaries) start here.

Expected ids are always derived via ``tok.index(...)`` - no hardcoded id
assumptions (the vocabulary order is an implementation detail, so it must never
leak into the tests).
"""

import dataclasses

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.lm.pairs import make_pairs
from orangecontrib.ai_suite.kernel.lm.tokenizer import (
    BOS,
    BOS_ID,
    EOS,
    EOS_ID,
    UNK,
    UNK_ID,
    CharTokenizer,
    Tokenizer,
    WordTokenizer,
    stream_units,
    tokenizer_from_dict,
    tokenizer_from_text,
    tokenizers,
)

CORPUS = "Hello world!\nDon't stop; it's o’clock.\nAB ab ab"


def test_special_ids_fixed():
    tok = tokenizer_from_text(CORPUS, "char")
    assert tok.token(BOS_ID) == BOS and tok.token(UNK_ID) == UNK
    assert tok.is_special(BOS_ID) and tok.is_special(UNK_ID)
    assert not tok.is_special(3)
    assert tok.counts[BOS_ID] == 0 and tok.counts[UNK_ID] == 0 and tok.counts[EOS_ID] == 0
    assert tok.token(EOS_ID) == EOS and tok.is_special(EOS_ID)


def test_tokenizers_catalogue():
    # The catalogue is derived from the classes, so it cannot go stale.
    catalogue = tokenizers()
    assert [cls.tokenizer_name() for cls, _ in catalogue] == ["char", "word"]
    assert all(desc for _, desc in catalogue)
    assert catalogue[0][1].startswith("Character tokenizer")
    assert catalogue[1][1].startswith("Word tokenizer")
    assert catalogue == tokenizers()  # stable across calls
    assert CharTokenizer.tokenizer_name() == "char"
    assert WordTokenizer.tokenizer_name() == "word"


def test_tokenizer_lookup_is_classes():
    assert isinstance(tokenizer_from_text(CORPUS, "char"), CharTokenizer)
    assert isinstance(tokenizer_from_text(CORPUS, "word"), WordTokenizer)
    with pytest.raises(TypeError):
        Tokenizer(tokens=(BOS, UNK), counts=(0, 0))  # abstract
    assert isinstance(tokenizer_from_text(CORPUS, "char"), CharTokenizer)
    assert isinstance(tokenizer_from_text(CORPUS, "word"), WordTokenizer)
    with pytest.raises(ValueError, match="tokenizer"):
        tokenizer_from_text(CORPUS, "sentencepiece")


def test_char_tokenizer_excludes_nothing():
    tok = tokenizer_from_text("a b", "char")
    ids = tok.encode("a b")
    assert len(ids) == 4  # 'a', ' ', 'b' + <eos>
    assert ids[-1] == EOS_ID
    assert tok.decode(ids) == "a b"
    assert tok.vocab_size == 3 + 3  # <bos>, <unk>, 'a', ' ', 'b'


def test_word_tokenizer_apostrophes_and_punctuation():
    tok = tokenizer_from_text(CORPUS, "word")
    tokens = [tok.token(i) for i in tok.encode("Don't stop; it's o’clock.")]
    assert tokens == ["Don't", "stop", ";", "it's", "o’clock", ".", EOS]


def test_word_tokenizer_preserves_case():
    tok = tokenizer_from_text("AB ab", "word")
    assert tok.index("AB") != tok.index("ab")
    assert tok.vocab_size == 3 + 2


def test_char_round_trip_exact_within_sentence():
    text = "Hello, world; done?"
    tok = tokenizer_from_text(text, "char")
    assert tok.decode(tok.encode(text)) == text


def test_char_sentence_separator_comes_back_as_one_space():
    # The separator is not a token, so it cannot survive verbatim: a newline
    # returns as a single space. Within a sentence nothing is invented.
    text = "Hello, world!\nSecond line; done?"
    tok = tokenizer_from_text(text, "char")
    assert tok.decode(tok.encode(text)) == "Hello, world! Second line; done?"


def test_word_round_trip_normalises_whitespace():
    tok = tokenizer_from_text(CORPUS, "word")
    out = tok.decode(tok.encode("Hello   world!\nDon't  stop."))
    assert out == "Hello world! Don't stop."


def test_word_round_trip_punctuation_spacing():
    # Closing punctuation attaches left, opening typographic quotes attach right.
    # The previous bug was curly quotes only: 'said “ hello ”' instead of 'said "hello"'.
    text = "He said “hello” to the ‘cat’s’ owner (x)"
    tok = tokenizer_from_text(text, "word")
    assert tok.decode(tok.encode(text)) == "He said “hello” to the ‘cat’s’ owner (x)"


def test_unknown_maps_to_unk():
    char_tok = tokenizer_from_text("ab", "char")
    word_tok = tokenizer_from_text("ab", "word")
    assert char_tok.encode("a¥")[-2] == UNK_ID  # [-1] is the unit's <eos>
    assert word_tok.encode("ab surprise")[-2] == UNK_ID
    assert UNK in word_tok.decode(word_tok.encode("¥"))


def test_counts_aligned_with_tokens():
    tok = tokenizer_from_text(CORPUS, "char")
    total = sum(1 for i in tok.encode(CORPUS) if i != EOS_ID)
    assert sum(tok.counts) == total
    assert tok.counts[0] == 0
    assert tok.total_token_count == total
    assert tok.dropped_occurrences == 0


# ---- encode emits sentence structure ---------------------------------------


def test_encode_ends_every_unit_with_eos():
    tok = tokenizer_from_text("AB. CD.", "char")
    A, B, C, D, DOT = (tok.index(c) for c in "ABCD.")
    assert tok.encode("AB. CD.") == [A, B, DOT, EOS_ID, C, D, DOT, EOS_ID]
    assert tok.encode("AB") == [A, B, EOS_ID]  # a unit always terminates
    assert tok.encode("") == []
    assert tok.encode("   \n") == []
    assert BOS_ID not in tok.encode("AB")  # starts are context, not data


def test_newline_separates_documents_in_one_text():
    # A corpus is one text: a newline between documents keeps their units apart,
    # so encode needs no corpus-level entry point of its own.
    tok = tokenizer_from_text("AB\nCD", "char")
    A, B, C, D = (tok.index(c) for c in "ABCD")
    assert tok.encode("AB\nCD") == [A, B, EOS_ID, C, D, EOS_ID]


# ---- order is undefined; only the aggregation is invariant -----------------


def test_counts_are_permutation_invariant():
    # Vocabulary *order* may differ; what a tokenizer reports must not.
    permuted = "\n".join(reversed(CORPUS.splitlines()))
    a = tokenizer_from_text(CORPUS, "word")
    b = tokenizer_from_text(permuted, "word")
    assert a.ranked_counts() == b.ranked_counts()
    assert a.vocab_size == b.vocab_size
    assert a.total_token_count == b.total_token_count


def test_rank_is_the_single_declared_order():
    # Ties break on the token, so a hapax cut is deterministic and corpus-order free.
    tok = tokenizer_from_text("bb aa cc\naa aa", "word")
    assert tok.ranked_counts() == [("aa", 3), ("bb", 1), ("cc", 1)]
    assert (
        tok.ranked_counts()[:2]
        == tokenizer_from_text("aa aa\ncc bb aa", "word").ranked_counts()[:2]
    )


def test_ranked_counts_excludes_specials():
    tok = tokenizer_from_text("a b", "char")
    assert [t for t, _ in tok.counted()] == ["a", " ", "b"]
    assert BOS not in [t for t, _ in tok.ranked_counts()]


# ---- pruning ---------------------------------------------------------------


def test_keep_top():
    corpus = "aa bb cc dd ee\naa aa bb bb"  # aa:3 bb:3 cc/dd/ee:1
    tok = tokenizer_from_text(corpus, "word")
    pruned = tok.keep_top(3)
    assert pruned.vocab_size == 3 + 3
    assert [t for t, _ in pruned.ranked_counts()] == ["aa", "bb", "cc"]  # tie alphabetical
    assert UNK_ID in pruned.encode("dd")  # least frequent -> <unk>
    assert UNK_ID not in pruned.encode("aa")


def test_drop_rare_by_count_and_by_share():
    corpus = "aa bb cc dd ee\naa aa bb bb"  # total 9 tokens
    tok = tokenizer_from_text(corpus, "word")
    assert [t for t, _ in tok.drop_rare(min_count=2).ranked_counts()] == ["aa", "bb"]
    assert [t for t, _ in tok.drop_rare(min_share=0.2).ranked_counts()] == ["aa", "bb"]


def test_shares_stay_corpus_relative_across_pruning():
    corpus = "aa bb cc dd ee\naa aa bb bb"  # total 9 tokens
    tok = tokenizer_from_text(corpus, "word")
    once = tok.drop_rare(min_count=2)
    twice = once.drop_rare(min_share=0.1)  # 0.1 * 9, not 0.1 * retained total
    assert twice.total_token_count == 9
    assert [t for t, _ in twice.ranked_counts()] == ["aa", "bb"]
    assert twice.dropped_occurrences == 9 - 6


def test_pruning_keeps_specials_and_returns_new_objects():
    tok = tokenizer_from_text("a b c", "word")
    pruned = tok.keep_top(1)
    assert pruned.tokens[:3] == (BOS, UNK, EOS)
    assert all(pruned.is_special(i) for i in range(3))
    assert tok.vocab_size == 6  # untouched: pruning is pure
    assert isinstance(pruned, WordTokenizer)


def test_pruning_error_paths():
    tok = tokenizer_from_text("aa bb cc", "word")
    with pytest.raises(ValueError, match="exactly one"):
        tok.drop_rare(min_count=2, min_share=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        tok.drop_rare()
    with pytest.raises(ValueError, match="min_count"):
        tok.drop_rare(min_count=0)
    with pytest.raises(ValueError, match="min_share"):
        tok.drop_rare(min_share=0.0)
    with pytest.raises(ValueError, match="n must be"):
        tok.keep_top(0)
    with pytest.raises(ValueError, match="removes every real token"):
        tok.drop_rare(min_count=99)


def test_frozen():
    tok = tokenizer_from_text(CORPUS, "char")
    with pytest.raises(dataclasses.FrozenInstanceError):
        tok.counts = (1,)


# ---- serialisation ---------------------------------------------------------


def test_dict_round_trip():
    tok = tokenizer_from_text(CORPUS, "word")
    again = tokenizer_from_dict(tok.to_dict())
    assert type(again) is WordTokenizer
    assert again.tokens == tok.tokens
    assert again.total_token_count == tok.total_token_count
    assert again.encode("Don't stop.") == tok.encode("Don't stop.")


def test_from_dict_preserves_saved_order():
    # Saved order is what ids were trained against, so it must never be re-sorted.
    data = {
        "tokenizer_name": "word",
        "tokens": [BOS, UNK, EOS, "z", "a"],
        "counts": [0, 0, 0, 1, 2],
        "total_token_count": 3,
    }
    tok = tokenizer_from_dict(data)
    assert tok.tokens == (BOS, UNK, EOS, "z", "a")
    assert tok.index("z") == 3
    assert tok.total_token_count == 3


def test_error_paths():
    with pytest.raises(ValueError, match="no tokens"):
        tokenizer_from_text("", "word")
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(tokens=(BOS, UNK, EOS, BOS), counts=(0, 0, 0, 1), total_token_count=1)
    with pytest.raises(ValueError, match="special tokens"):
        CharTokenizer(tokens=("a", UNK, EOS), counts=(1, 0, 0), total_token_count=1)
    with pytest.raises(ValueError, match="same length"):
        CharTokenizer(tokens=(BOS, UNK, EOS, "a"), counts=(0, 0, 0), total_token_count=0)


# ---- unit structure: one spelling, shared by decode and make_pairs --------


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ([], [[]]),  # no ids: one empty unit, and make_pairs refuses it earlier
        ([EOS_ID], [[], []]),  # one terminated unit, then the empty tail
        ([3, 4, EOS_ID], [[3, 4], []]),
        ([3, EOS_ID, 4, EOS_ID], [[3], [4], []]),
        ([3, EOS_ID, EOS_ID], [[3], [], []]),  # a doubled EOS is an empty unit
        ([3, EOS_ID, 4], [[3], [4]]),  # unterminated: the tail is still a unit
        ([BOS_ID, 3, EOS_ID], [[3], []]),  # BOS is context, never content
        ([3, BOS_ID, 4, EOS_ID], [[3, 4], []]),  # ... anywhere in the unit
        (np.array([3, 4, EOS_ID]), [[3, 4], []]),  # any iterable of ids
    ],
)
def test_stream_units_splits_at_the_eos_boundaries(stream, expected):
    assert stream_units(stream) == expected


def test_the_two_consumers_see_the_same_units():
    # decode and make_pairs both derive unit structure from this one function,
    # so a stream of two sentences is two units for each of them.
    tok = tokenizer_from_text("AB. CD.", "char")
    stream = tok.encode("AB. CD.")
    assert len(stream_units(stream)) - 1 == 2  # the trailing empty tail, dropped
    assert len(make_pairs(stream, context_width=2).pairs_per_unit) == 2
    assert tok.decode(stream) == "AB. CD."


def test_decode_keeps_unterminated_text_and_drops_bos():
    tok = tokenizer_from_text("AB. CD.", "char")
    ids = tok.encode("AB.")  # ends with EOS
    assert tok.decode([BOS_ID, *ids[:-1]]) == "AB."  # no EOS: still rendered
