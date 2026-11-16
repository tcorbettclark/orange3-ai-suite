"""Tokenizer reporting tests: the read-only views a widget renders.

These live on the tokenizer (``stats``, ``frequency_table``, ``zipf_summary``)
and never feed back into training.
"""

import pytest

from orangecontrib.ai_suite.kernel.lm import Tokenizer

CORPUS = [
    "the cat sat on the mat",
    "the cat saw the dog",
    "a dog and a cat",
]


@pytest.fixture(scope="module")
def word_tok():
    return Tokenizer.from_corpus(CORPUS, "word")


@pytest.fixture(scope="module")
def char_tok():
    return Tokenizer.from_corpus(CORPUS, "char")


def test_frequency_table_ranks(word_tok):
    table = word_tok.frequency_table(n=4)
    ranks = [r for r, _, _ in table]
    assert ranks == [1, 2, 3, 4]
    assert table[0][1] == "the" and table[0][2] == 4
    assert table[1][1] == "cat" and table[1][2] == 3
    # count-2 tie {a, dog} broken alphabetically: 'a' then 'dog'
    assert table[2][1] == "a" and table[2][2] == 2
    assert table[3][1] == "dog" and table[3][2] == 2


def test_zipf_summary(word_tok):
    z = word_tok.zipf_summary()
    assert z["count_top1"] == 4
    assert 0 < z["top1_share"] < 1
    assert z["zipf_product_mean"] > 0
    assert z["zipf_product_spread"] >= 0


def test_stats(word_tok):
    s = word_tok.stats()
    assert s["level"] == "word"
    assert s["vocab_size"] == word_tok.vocab_size
    assert s["special_tokens"] == ("<bos>", "<unk>")
    longest = s["longest_tokens"]
    assert all(len(longest[i][0]) >= len(longest[i + 1][0]) for i in range(len(longest) - 1))
    assert 0.0 <= s["single_occurrence_fraction"] <= 1.0
    assert s["mean_token_length"] > 0
    assert s["dropped_occurrences"] == 0


def test_small_vocab_char_vs_word_is_the_lesson():
    # A word-rich but alphabet-poor corpus: the whole point of the LM Pre-trainer's
    # tokenizer dropdown — char level = far smaller vocab, longer sequences.
    words = " ".join(a + b for a in "abcd" for b in "abcdef")
    char_tok = Tokenizer.from_corpus([words], "char")
    word_tok = Tokenizer.from_corpus([words], "word")
    assert char_tok.stats()["vocab_size"] < word_tok.stats()["vocab_size"]
    text = "aa ab ac"
    assert len(char_tok.encode(text)) > len(word_tok.encode(text))


def test_longest_tokens_ties_are_deterministic():
    # Equal-length hapaxes used to inherit vocabulary order, which is undefined;
    # the view must break the tie itself.
    corpus = ["abc abd xyz"]
    a = Tokenizer.from_corpus(corpus, "word").stats()["longest_tokens"]
    b = Tokenizer.from_corpus(["abd xyz abc"], "word").stats()["longest_tokens"]
    assert a == b == [("abc", 1), ("abd", 1), ("xyz", 1)]


def test_shares_are_of_the_corpus_not_the_retained_vocab():
    # The Zipf lesson must not change when the vocabulary is pruned.
    tok = Tokenizer.from_corpus(["the cat sat", "the cat saw the dog"], "word")
    pruned = tok.keep_top(2)
    assert pruned.total_token_count == tok.total_token_count
    assert pruned.zipf_summary()["top1_share"] == pytest.approx(tok.zipf_summary()["top1_share"])
    assert pruned.stats()["dropped_occurrences"] > 0


def test_reporting_methods_are_read_only(word_tok):
    before = word_tok.to_dict()
    word_tok.stats()
    word_tok.frequency_table()
    word_tok.zipf_summary()
    assert word_tok.to_dict() == before


def test_empty_zipf_summary():
    # Only specials left: reporting must not divide by zero.
    tok = Tokenizer.from_corpus(["a"], "word")
    assert tok.zipf_summary(top=0) == {"n": 0}
    assert tok.stats()["mean_token_length"] == pytest.approx(1.0)
