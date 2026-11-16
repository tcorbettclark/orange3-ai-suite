"""Tokenizer edge cases: special ids, levels, encode/decode round trips, pruning
(BOS/padding, unknowns, sentence boundaries) start here.

Expected ids are always derived via ``tok.index(...)`` - no hardcoded id
assumptions (the vocabulary order is an implementation detail, so it must never
leak into the tests).
"""

import dataclasses

import pytest

from orangecontrib.ai_suite.kernel.lm import (
    BOS,
    BOS_ID,
    UNK,
    UNK_ID,
    CharTokenizer,
    Tokenizer,
    WordTokenizer,
)

CORPUS = ["Hello world!", "Don't stop; it's o’clock.", "AB ab ab"]


def test_special_ids_fixed():
    tok = Tokenizer.from_corpus(CORPUS, "char")
    assert tok.token(BOS_ID) == BOS and tok.token(UNK_ID) == UNK
    assert tok.is_special(BOS_ID) and tok.is_special(UNK_ID)
    assert not tok.is_special(2)
    assert tok.counts[BOS_ID] == 0 and tok.counts[UNK_ID] == 0


def test_levels_catalogue():
    # The catalogue is derived from the classes, so it cannot go stale.
    levels = Tokenizer.levels()
    assert [name for name, _ in levels] == ["char", "word"]
    assert all(desc for _, desc in levels)
    assert levels[0][1].startswith("Character level")
    assert levels[1][1].startswith("Word level")
    assert levels == Tokenizer.levels()  # stable across calls


def test_levels_are_classes():
    assert isinstance(Tokenizer.from_corpus(CORPUS, "char"), CharTokenizer)
    assert isinstance(Tokenizer.from_corpus(CORPUS, "word"), WordTokenizer)
    with pytest.raises(TypeError):
        Tokenizer(tokens=(BOS, UNK), counts=(0, 0))  # abstract
    with pytest.raises(ValueError, match="level"):
        Tokenizer.from_corpus(CORPUS, "sentencepiece")


def test_char_level_excludes_nothing():
    tok = Tokenizer.from_corpus(["a b"], "char")
    ids = tok.encode("a b")
    assert ids[0] == BOS_ID
    assert len(ids) == 4  # <bos> + 'a', ' ', 'b'
    assert tok.decode(ids) == "a b"
    assert tok.vocab_size == 2 + 3  # <bos>, <unk>, 'a', ' ', 'b'


def test_word_level_apostrophes_and_punctuation():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    tokens = [tok.token(i) for i in tok.encode("Don't stop; it's o’clock.")]
    assert tokens == [BOS, "Don't", "stop", ";", "it's", "o’clock", "."]


def test_word_level_preserves_case():
    tok = Tokenizer.from_corpus(["AB ab"], "word")
    assert tok.index("AB") != tok.index("ab")
    assert tok.vocab_size == 2 + 2


def test_char_round_trip_exact_within_sentence():
    text = "Hello, world; done?"
    tok = Tokenizer.from_corpus([text], "char")
    assert tok.decode(tok.encode(text)) == text


def test_char_sentence_separator_comes_back_as_one_space():
    # The separator is not a token, so it cannot survive verbatim: a newline
    # returns as a single space. Within a sentence nothing is invented.
    text = "Hello, world!\nSecond line; done?"
    tok = Tokenizer.from_corpus([text], "char")
    assert tok.decode(tok.encode(text)) == "Hello, world! Second line; done?"


def test_word_round_trip_normalises_whitespace():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    out = tok.decode(tok.encode("Hello   world!\nDon't  stop."))
    assert out == "Hello world! Don't stop."


def test_word_round_trip_punctuation_spacing():
    # Closing punctuation attaches left, opening typographic quotes attach right.
    # The previous bug was curly quotes only: 'said “ hello ”' instead of 'said "hello"'.
    text = "He said “hello” to the ‘cat’s’ owner (x)"
    tok = Tokenizer.from_corpus([text], "word")
    assert tok.decode(tok.encode(text)) == "He said “hello” to the ‘cat’s’ owner (x)"


def test_unknown_maps_to_unk():
    char_tok = Tokenizer.from_corpus(["ab"], "char")
    word_tok = Tokenizer.from_corpus(["ab"], "word")
    assert char_tok.encode("a¥")[-1] == UNK_ID
    assert word_tok.encode("ab surprise")[-1] == UNK_ID
    assert UNK in word_tok.decode(word_tok.encode("¥"))


def test_counts_aligned_with_tokens():
    tok = Tokenizer.from_corpus(CORPUS, "char")
    total = sum(1 for t in CORPUS for i in tok.encode(t) if i != BOS_ID)
    assert sum(tok.counts) == total
    assert tok.counts[0] == 0
    assert tok.total_token_count == total
    assert tok.dropped_occurrences == 0


# ---- encode emits sentence structure ---------------------------------------


def test_encode_marks_every_sentence():
    tok = Tokenizer.from_corpus(["AB. CD."], "char")
    A, B, C, D, DOT = (tok.index(c) for c in "ABCD.")
    assert tok.encode("AB. CD.") == [BOS_ID, A, B, DOT, BOS_ID, C, D, DOT]
    assert tok.encode("AB") == [BOS_ID, A, B]  # single sentence is still marked
    assert tok.encode("") == []
    assert tok.encode("   \n") == []
    assert BOS not in tok.encode("AB")[1:]  # BOS only at sentence starts


def test_encode_corpus_concatenates_documents():
    tok = Tokenizer.from_corpus(["AB", "CD"], "char")
    A, B, C, D = (tok.index(c) for c in "ABCD")
    assert tok.encode_corpus(["AB", "CD"]) == [BOS_ID, A, B, BOS_ID, C, D]
    assert tok.encode_corpus([]) == []
    assert tok.encode_corpus(["AB"]) == tok.encode("AB")


# ---- order is undefined; only the aggregation is invariant -----------------


def test_counts_are_permutation_invariant():
    # Vocabulary *order* may differ; what a tokenizer reports must not.
    a = Tokenizer.from_corpus(CORPUS, "word")
    b = Tokenizer.from_corpus(reversed(CORPUS), "word")
    assert a.ranked_counts() == b.ranked_counts()
    assert a.vocab_size == b.vocab_size
    assert a.total_token_count == b.total_token_count


def test_rank_is_the_single_declared_order():
    # Ties break on the token, so a hapax cut is deterministic and corpus-order free.
    tok = Tokenizer.from_corpus(["bb aa cc", "aa aa"], "word")
    assert tok.ranked_counts() == [("aa", 3), ("bb", 1), ("cc", 1)]
    assert (
        tok.ranked_counts()[:2]
        == Tokenizer.from_corpus(["aa aa", "cc bb aa"], "word").ranked_counts()[:2]
    )


def test_ranked_counts_excludes_specials():
    tok = Tokenizer.from_corpus(["a b"], "char")
    assert [t for t, _ in tok.counted()] == ["a", " ", "b"]
    assert BOS not in [t for t, _ in tok.ranked_counts()]


# ---- pruning ---------------------------------------------------------------


def test_keep_top():
    corpus = ["aa bb cc dd ee", "aa aa bb bb"]  # aa:3 bb:3 cc/dd/ee:1
    tok = Tokenizer.from_corpus(corpus, "word")
    pruned = tok.keep_top(3)
    assert pruned.vocab_size == 3 + 2
    assert [t for t, _ in pruned.ranked_counts()] == ["aa", "bb", "cc"]  # tie alphabetical
    assert UNK_ID in pruned.encode("dd")  # least frequent -> <unk>
    assert UNK_ID not in pruned.encode("aa")


def test_drop_rare_by_count_and_by_share():
    corpus = ["aa bb cc dd ee", "aa aa bb bb"]  # total 9 tokens
    tok = Tokenizer.from_corpus(corpus, "word")
    assert [t for t, _ in tok.drop_rare(min_count=2).ranked_counts()] == ["aa", "bb"]
    assert [t for t, _ in tok.drop_rare(min_share=0.2).ranked_counts()] == ["aa", "bb"]


def test_shares_stay_corpus_relative_across_pruning():
    corpus = ["aa bb cc dd ee", "aa aa bb bb"]  # total 9 tokens
    tok = Tokenizer.from_corpus(corpus, "word")
    once = tok.drop_rare(min_count=2)
    twice = once.drop_rare(min_share=0.1)  # 0.1 * 9, not 0.1 * retained total
    assert twice.total_token_count == 9
    assert [t for t, _ in twice.ranked_counts()] == ["aa", "bb"]
    assert twice.dropped_occurrences == 9 - 6


def test_pruning_keeps_specials_and_returns_new_objects():
    tok = Tokenizer.from_corpus(["a b c"], "word")
    pruned = tok.keep_top(1)
    assert pruned.tokens[:2] == (BOS, UNK)
    assert pruned.is_special(BOS_ID) and pruned.is_special(UNK_ID)
    assert tok.vocab_size == 5  # untouched: pruning is pure
    assert isinstance(pruned, WordTokenizer)


def test_pruning_error_paths():
    tok = Tokenizer.from_corpus(["aa bb cc"], "word")
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
    tok = Tokenizer.from_corpus(CORPUS, "char")
    with pytest.raises(dataclasses.FrozenInstanceError):
        tok.counts = (1,)


# ---- serialisation ---------------------------------------------------------


def test_dict_round_trip():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    again = Tokenizer.from_dict(tok.to_dict())
    assert type(again) is WordTokenizer
    assert again.tokens == tok.tokens
    assert again.total_token_count == tok.total_token_count
    assert again.encode("Don't stop.") == tok.encode("Don't stop.")


def test_from_dict_preserves_order_and_tolerates_old_saves():
    # Saved order is what ids were trained against, so it must never be re-sorted.
    data = {"level": "word", "tokens": [BOS, UNK, "z", "a"], "counts": [0, 0, 1, 2]}
    tok = Tokenizer.from_dict(data)
    assert tok.tokens == (BOS, UNK, "z", "a")
    assert tok.index("z") == 2
    assert tok.total_token_count == 3  # absent key -> retained sum


def test_error_paths():
    with pytest.raises(ValueError, match="no tokens"):
        Tokenizer.from_corpus(["", "   \n"], "word")
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(tokens=(BOS, UNK, BOS), counts=(0, 0, 1))
    with pytest.raises(ValueError, match="special tokens"):
        CharTokenizer(tokens=("a", UNK), counts=(1, 0))
    with pytest.raises(ValueError, match="same length"):
        CharTokenizer(tokens=(BOS, UNK, "a"), counts=(0, 0))
