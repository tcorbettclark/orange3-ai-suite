"""Tokenizer edge cases: special ids, levels, encode/decode round trips.
(BOS/padding, unknowns, sentence boundaries) start here.

Expected ids are always derived via ``tok.index(...)`` — no hardcoded id
assumptions (alphabetical tie-breaks would otherwise leak into the tests).
"""

import pytest

from nn_kernel.lm import BOS, BOS_ID, UNK, UNK_ID, Tokenizer

CORPUS = ["Hello world!", "Don't stop; it's o’clock.", "AB ab ab"]


def test_special_ids_fixed():
    tok = Tokenizer.from_corpus(CORPUS, "char")
    assert tok.token(BOS_ID) == BOS and tok.token(UNK_ID) == UNK
    assert tok.is_special(BOS_ID) and tok.is_special(UNK_ID)
    assert not tok.is_special(2)
    assert tok.counts[BOS_ID] == 0 and tok.counts[UNK_ID] == 0


def test_char_level_excludes_nothing():
    tok = Tokenizer.from_corpus(["a b"], "char")
    ids = tok.encode("a b")
    assert len(ids) == 3  # the space is a real token
    assert tok.decode(ids) == "a b"
    assert tok.vocab_size == 2 + 3  # <bos>, <unk>, 'a', ' ', 'b'


def test_word_level_apostrophes_and_punctuation():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    tokens = [tok.token(i) for i in tok.encode("Don't stop; it's o’clock.")]
    assert tokens == ["Don't", "stop", ";", "it's", "o’clock", "."]


def test_word_level_preserves_case():
    tok = Tokenizer.from_corpus(["AB ab"], "word")
    assert tok.index("AB") != tok.index("ab")
    assert tok.vocab_size == 2 + 2


def test_char_round_trip_exact():
    text = "Hello, world!\nSecond line; done?"
    tok = Tokenizer.from_corpus([text], "char")
    assert tok.decode(tok.encode(text)) == text


def test_word_round_trip_normalises_whitespace():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    out = tok.decode(tok.encode("Hello   world!\nDon't  stop."))
    assert out == "Hello world! Don't stop."


def test_unknown_maps_to_unk():
    char_tok = Tokenizer.from_corpus(["ab"], "char")
    word_tok = Tokenizer.from_corpus(["ab"], "word")
    assert char_tok.encode("a¥")[-1] == UNK_ID
    assert word_tok.encode("ab surprise")[-1] == UNK_ID
    assert UNK in word_tok.decode(word_tok.encode("¥"))


def test_max_vocab_truncates_least_frequent():
    corpus = ["aa bb cc dd ee", "aa aa bb bb"]  # aa:4 bb:3 cc/dd/ee:1
    tok = Tokenizer.from_corpus(corpus, "word", max_vocab=5)
    assert tok.vocab_size == 5
    assert [tok.token(i) for i in range(2, 5)] == ["aa", "bb", "cc"]  # ties alphabetical
    assert UNK_ID in tok.encode("dd")  # least frequent -> <unk>
    assert UNK_ID not in tok.encode("aa")


def test_deterministic_and_order_independent():
    a = Tokenizer.from_corpus(CORPUS, "char")
    b = Tokenizer.from_corpus(reversed(CORPUS), "char")
    assert a.tokens == b.tokens and a.counts == b.counts
    assert (
        Tokenizer.from_corpus(CORPUS, "word").tokens == Tokenizer.from_corpus(CORPUS, "word").tokens
    )


def test_counts_aligned_with_tokens():
    tok = Tokenizer.from_corpus(CORPUS, "char")
    total = sum(len(tok.encode(t)) for t in CORPUS)
    assert sum(tok.counts) == total
    assert tok.counts[0] == 0


def test_error_paths():
    with pytest.raises(ValueError, match="level"):
        Tokenizer.from_corpus(CORPUS, "sentencepiece")
    with pytest.raises(ValueError, match="no tokens"):
        Tokenizer.from_corpus(["", "   \n"], "word")
    with pytest.raises(ValueError, match="max_vocab"):
        Tokenizer.from_corpus(CORPUS, "char", max_vocab=1)
    with pytest.raises(ValueError, match="duplicate"):
        Tokenizer(level="char", tokens=(BOS, UNK, BOS), counts=(0, 0, 1))


def test_dict_round_trip():
    tok = Tokenizer.from_corpus(CORPUS, "word")
    again = Tokenizer.from_dict(tok.to_dict())
    assert again.tokens == tok.tokens
    assert again.encode("Don't stop.") == tok.encode("Don't stop.")
