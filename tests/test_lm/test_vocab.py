"""Vocab Inspector statistics tests (read-only logic)."""

import pytest

from nn_kernel.lm import Tokenizer, frequency_table, side_by_side, vocab_stats, zipf_summary


@pytest.fixture(scope="module")
def word_tok():
    corpus = [
        "the cat sat on the mat",
        "the cat saw the dog",
        "a dog and a cat",
    ]
    return Tokenizer.from_corpus(corpus, "word")


@pytest.fixture(scope="module")
def char_tok():
    corpus = [
        "the cat sat on the mat",
        "the cat saw the dog",
        "a dog and a cat",
    ]
    return Tokenizer.from_corpus(corpus, "char")


def test_frequency_table_ranks(word_tok):
    table = frequency_table(word_tok, n=4)
    ranks = [r for r, _, _ in table]
    assert ranks == [1, 2, 3, 4]
    assert table[0][1] == "the" and table[0][2] == 4
    assert table[1][1] == "cat" and table[1][2] == 3
    # count-2 tie {a, dog} broken alphabetically: 'a' then 'dog'
    assert table[2][1] == "a" and table[2][2] == 2
    assert table[3][1] == "dog" and table[3][2] == 2


def test_zipf_summary(word_tok):
    z = zipf_summary(word_tok)
    assert z["count_top1"] == 4
    assert 0 < z["top1_share"] < 1
    assert z["zipf_product_mean"] > 0
    assert z["zipf_product_spread"] >= 0


def test_vocab_stats(word_tok):
    s = vocab_stats(word_tok)
    assert s["level"] == "word"
    assert s["vocab_size"] == word_tok.vocab_size
    assert s["special_tokens"] == ("<bos>", "<unk>")
    longest = s["longest_tokens"]
    assert all(len(longest[i][0]) >= len(longest[i + 1][0]) for i in range(len(longest) - 1))
    assert 0.0 <= s["single_occurrence_fraction"] <= 1.0
    assert s["mean_token_length"] > 0


def test_small_vocab_char_vs_word_is_the_lesson():
    # A word-rich but alphabet-poor corpus: the whole point of the Trainer's
    # tokenizer dropdown — char level = far smaller vocab, longer sequences.
    words = " ".join(a + b for a in "abcd" for b in "abcdef")
    char_tok = Tokenizer.from_corpus([words], "char")
    word_tok = Tokenizer.from_corpus([words], "word")
    assert vocab_stats(char_tok)["vocab_size"] < vocab_stats(word_tok)["vocab_size"]
    text = "aa ab ac"
    assert len(char_tok.encode(text)) > len(word_tok.encode(text))


def test_side_by_side(char_tok, word_tok):
    view = side_by_side(char_tok, word_tok, "a cat")
    assert [t for t, _ in view["char"]] == ["a", " ", "c", "a", "t"]
    assert [t for t, _ in view["word"]] == ["a", "cat"]
    assert view["word"][0][1] == word_tok.index("a")


def test_side_by_side_level_guard(word_tok, char_tok):
    with pytest.raises(ValueError, match="expects"):
        side_by_side(word_tok, word_tok, "text")
    with pytest.raises(ValueError, match="expects"):
        side_by_side(char_tok, char_tok, "text")
