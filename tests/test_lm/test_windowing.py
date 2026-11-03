"""Windowing tests — BOS/padding semantics and sentence-boundary discipline.

Expected token ids are derived via ``tok.index(...)``; nothing assumes the
vocabulary's internal sort order.
"""

import numpy as np
import pytest

from nn_kernel.lm import Tokenizer, make_windows, split_sentences
from nn_kernel.lm.windowing import Windows


@pytest.fixture(scope="module")
def char_tok():
    return Tokenizer.from_corpus(["ABCD.", "EFGHI.", "AB. CD."], "char")


def test_sentence_splitting_basics():
    assert split_sentences("Hi! You are... ok? Yes.") == [
        "Hi!",
        "You are...",
        "ok?",
        "Yes.",
    ]
    assert split_sentences("AB.\nCD.") == ["AB.", "CD."]  # newline = hard break
    assert split_sentences("no terminator here") == ["no terminator here"]
    assert split_sentences("What?! No.") == ["What?!", "No."]
    assert split_sentences("") == []
    assert split_sentences("Dr. Who") == ["Dr.", "Who"]  # accepted limitation


def test_bos_padding_sequence_start(char_tok):
    A, B, DOT = (char_tok.index(c) for c in "AB.")
    windows = make_windows(char_tok, ["AB."], context_size=3)
    # sentence ids: A B . ; seq = [BOS, A, B, .]
    # i=1: [B,B,B] -> A ; i=2: [B,B,A] -> B ; i=3: [B,A,B] -> .
    assert np.array_equal(
        windows.contexts, np.array([[0, 0, 0], [0, 0, A], [0, A, B]], dtype=np.int32)
    )
    assert np.array_equal(windows.targets, np.array([A, B, DOT], dtype=np.int32))
    assert windows.sentence_lengths == (3,)


def test_windows_never_cross_sentences(char_tok):
    A, C, E = (char_tok.index(c) for c in "ACE")
    windows = make_windows(char_tok, ["AB.", "CD."], context_size=3)
    assert len(windows) == 6
    assert windows.sentence_lengths == (3, 3)
    # sentence-2 windows never mention sentence-1-only letters and vice versa
    assert A not in windows.contexts[3:].ravel()
    assert C not in windows.contexts[:3].ravel()
    assert E not in windows.contexts.ravel()  # EFGHI wasn't in this corpus
    # the first context of sentence 2 is all-BOS, not a tail of sentence 1
    assert np.array_equal(windows.contexts[3], [0, 0, 0])
    # sentence 2 predicts exactly its own tokens
    assert np.array_equal(
        windows.targets[3:],
        np.array([char_tok.index(c) for c in "CD."], dtype=np.int32),
    )


def test_window_sliding_within_sentence(char_tok):
    BOS, A, B, C, D, DOT = (
        0,
        char_tok.index("A"),
        char_tok.index("B"),
        char_tok.index("C"),
        char_tok.index("D"),
        char_tok.index("."),
    )
    windows = make_windows(char_tok, ["ABCD."], context_size=2)
    # seq [BOS A B C D .] with k=2:
    # [B B]->A, [B A]->B, [A B]->C, [B C]->D, [C D]->.
    assert np.array_equal(
        windows.contexts,
        np.array([[BOS, BOS], [BOS, A], [A, B], [B, C], [C, D]], dtype=np.int32),
    )
    assert np.array_equal(windows.targets, np.array([A, B, C, D, DOT], dtype=np.int32))


def test_large_context_all_bos_fill(char_tok):
    A, B = (char_tok.index(c) for c in "AB")
    windows = make_windows(char_tok, ["AB."], context_size=10)
    assert windows.contexts.shape == (3, 10)
    # window for position 3: 7 BOS pads + the 3 preceding tokens [BOS, A, B]
    assert np.array_equal(windows.contexts[2, :7], np.zeros(7, dtype=np.int32))
    assert np.array_equal(windows.contexts[2, 7:], np.array([0, A, B], dtype=np.int32))
    assert windows.targets[2] == char_tok.index(".")


def test_context_size_one(char_tok):
    A, B, DOT = (char_tok.index(c) for c in "AB.")
    windows = make_windows(char_tok, ["AB."], context_size=1)
    assert np.array_equal(windows.contexts.ravel(), [0, A, B])
    assert np.array_equal(windows.targets, [A, B, DOT])


def test_empty_sentences_skipped(char_tok):
    real = "...real sentence here."
    windows = make_windows(char_tok, [real, "   "], context_size=2)
    assert windows.sentence_lengths == (len(real),)  # len == char tokens


def test_sentence_split_happens_inside(char_tok):
    windows = make_windows(char_tok, ["AB. CD."], context_size=1)
    assert windows.sentence_lengths == (3, 3)


def test_no_windows_raises(char_tok):
    with pytest.raises(ValueError, match="no windows"):
        make_windows(char_tok, ["", "  \n"], context_size=2)
    with pytest.raises(ValueError, match="context_size"):
        make_windows(char_tok, ["AB."], context_size=0)


def test_split_is_sentence_disjoint_deterministic_and_complete():
    docs = [f"{c}{c}." for c in "abcdefghijkl"]  # 12 sentences, shared letters
    tok = Tokenizer.from_corpus(docs, "char")
    windows = make_windows(tok, docs, context_size=2)

    train, val = windows.split(val_fraction=0.25, seed=1)
    assert len(val.sentence_lengths) == max(1, round(12 * 0.25)) == 3
    assert sum(train.sentence_lengths) == len(train)
    assert sum(val.sentence_lengths) == len(val)
    assert len(train) + len(val) == len(windows)

    # complete partition: as multisets, (context, target) rows are preserved
    def multiset(w):
        pairs = [tuple(r) + (int(t),) for r, t in zip(w.contexts, w.targets)]
        return sorted(pairs)

    assert multiset(windows) == sorted(multiset(train) + multiset(val))

    # deterministic under a fixed seed
    t2, v2 = windows.split(val_fraction=0.25, seed=1)
    assert np.array_equal(train.targets, t2.targets)
    assert np.array_equal(val.contexts, v2.contexts)


def test_split_fraction_validation():
    tok = Tokenizer.from_corpus(["ab.", "cd."], "char")
    windows = make_windows(tok, ["ab.", "cd."], context_size=1)
    with pytest.raises(ValueError, match="val_fraction"):
        windows.split(val_fraction=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        make_windows(tok, ["ab."], context_size=1).split()


def test_windows_consistency_guard():
    with pytest.raises(ValueError, match="sentence_lengths"):
        Windows(
            contexts=np.zeros((4, 2), dtype=np.int32),
            targets=np.zeros(4, dtype=np.int32),
            context_size=2,
            sentence_lengths=(3, 3),
        )