"""Windowing tests — BOS/padding semantics and sentence-boundary discipline.

``make_windows`` takes a BOS-marked token stream, so every case is tested on
hand-written integers: no vocabulary, no corpus, no tokenizer. The expected
arrays are the ones the old text-taking implementation produced, which is what
makes this a refactor rather than a change of behaviour.
"""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.lm import Tokenizer, make_windows
from orangecontrib.ai_suite.kernel.lm.windowing import Windows

BOS = 0


# ---- window arithmetic -----------------------------------------------------


def test_bos_padding_sequence_start():
    # [BOS, 2, 3, 4] with k=3: [B,B,B]->2, [B,B,2]->3, [B,2,3]->4
    windows = make_windows([BOS, 2, 3, 4], context_size=3)
    assert np.array_equal(
        windows.contexts, np.array([[0, 0, 0], [0, 0, 2], [0, 2, 3]], dtype=np.int32)
    )
    assert np.array_equal(windows.targets, np.array([2, 3, 4], dtype=np.int32))
    assert windows.sentence_lengths == (3,)
    assert windows.context_size == 3


def test_windows_never_cross_sentences():
    windows = make_windows([BOS, 2, 3, 4, BOS, 5, 6, 7], context_size=3)
    assert len(windows) == 6
    assert windows.sentence_lengths == (3, 3)
    # no id from one sentence appears in the other's windows
    assert not set(windows.contexts[3:].ravel()) & {2, 3, 4}
    assert not set(windows.contexts[:3].ravel()) & {5, 6, 7}
    # the first context of sentence 2 is all-BOS, not a tail of sentence 1
    assert np.array_equal(windows.contexts[3], [BOS, BOS, BOS])
    assert np.array_equal(windows.targets[3:], np.array([5, 6, 7], dtype=np.int32))


def test_window_sliding_within_sentence():
    windows = make_windows([BOS, 2, 3, 4, 5, 6], context_size=2)
    # [B B]->2, [B 2]->3, [2 3]->4, [3 4]->5, [4 5]->6
    assert np.array_equal(
        windows.contexts,
        np.array([[0, 0], [0, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32),
    )
    assert np.array_equal(windows.targets, np.array([2, 3, 4, 5, 6], dtype=np.int32))


def test_large_context_all_bos_fill():
    windows = make_windows([BOS, 2, 3, 4], context_size=10)
    assert windows.contexts.shape == (3, 10)
    # window for the third target: 7 BOS pads + [BOS, 2, 3]
    assert np.array_equal(windows.contexts[2, :7], np.zeros(7, dtype=np.int32))
    assert np.array_equal(windows.contexts[2, 7:], np.array([0, 2, 3], dtype=np.int32))
    assert windows.targets[2] == 4


def test_context_size_one():
    windows = make_windows([BOS, 2, 3, 4], context_size=1)
    assert np.array_equal(windows.contexts.ravel(), [0, 2, 3])
    assert np.array_equal(windows.targets, [2, 3, 4])


def test_single_token_sentence():
    windows = make_windows([BOS, 2, BOS, 3], context_size=2)
    assert windows.sentence_lengths == (1, 1)
    assert np.array_equal(windows.contexts, [[0, 0], [0, 0]])
    assert np.array_equal(windows.targets, [2, 3])


def test_bos_is_never_a_target():
    windows = make_windows([BOS, 2, 3, 4, BOS, 5], context_size=2)
    assert BOS not in windows.targets  # generation relies on this


def test_error_paths():
    with pytest.raises(ValueError, match="context_size"):
        make_windows([BOS, 2], context_size=0)
    with pytest.raises(ValueError, match="must start with <bos>"):
        make_windows([], context_size=2)
    with pytest.raises(ValueError, match="must start with <bos>"):
        make_windows([2, 3, 4], context_size=2)  # unmarked stream
    with pytest.raises(ValueError, match="empty sentence"):
        make_windows([BOS, BOS, 2], context_size=2)
    with pytest.raises(ValueError, match="empty sentence"):
        make_windows([BOS, 2, BOS], context_size=2)  # trailing marker, no tokens


# ---- end to end: an encoded corpus is a self-describing stream -------------


def test_encoded_corpus_needs_no_text():
    tok = Tokenizer.from_corpus(["AB.", "CD."], "char")
    stream = tok.encode_corpus(["AB.", "CD."])
    assert stream == tok.encode("AB.") + tok.encode("CD.")
    windows = make_windows(stream, context_size=3)
    assert windows.sentence_lengths == (3, 3)
    assert len(windows) == 6
    assert np.array_equal(windows.contexts[3], [BOS, BOS, BOS])


def test_documents_are_not_merged_into_one_sentence():
    # "AB" has no terminator: concatenating the *texts* would fuse the two
    # documents, concatenating their streams cannot.
    tok = Tokenizer.from_corpus(["AB", "CD"], "char")
    windows = make_windows(tok.encode_corpus(["AB", "CD"]), context_size=1)
    assert windows.sentence_lengths == (2, 2)


# ---- splitting -------------------------------------------------------------


def twelve_sentences() -> Windows:
    stream = [x for i in range(12) for x in (BOS, i + 2, i + 3)]
    return make_windows(stream, context_size=2)


def test_split_is_sentence_disjoint_deterministic_and_complete():
    windows = twelve_sentences()
    train, val = windows.split(val_fraction=0.25, seed=1)
    assert len(val.sentence_lengths) == max(1, round(12 * 0.25)) == 3
    assert sum(train.sentence_lengths) == len(train)
    assert sum(val.sentence_lengths) == len(val)
    assert len(train) + len(val) == len(windows)

    # complete partition: as multisets, (context, target) rows are preserved
    def multiset(w):
        pairs = [tuple(r) + (int(t),) for r, t in zip(w.contexts, w.targets, strict=True)]
        return sorted(pairs)

    assert multiset(windows) == sorted(multiset(train) + multiset(val))

    # deterministic under a fixed seed
    t2, v2 = windows.split(val_fraction=0.25, seed=1)
    assert np.array_equal(train.targets, t2.targets)
    assert np.array_equal(val.contexts, v2.contexts)


def test_split_fraction_validation():
    windows = make_windows([BOS, 2, 3, BOS, 4, 5], context_size=1)
    with pytest.raises(ValueError, match="val_fraction"):
        windows.split(val_fraction=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        make_windows([BOS, 2, 3], context_size=1).split()


def test_split_caps_validation_to_leave_training_rows():
    windows = make_windows([BOS, 2, 3, BOS, 4, 5], context_size=1)
    train, val = windows.split(val_fraction=0.99)
    assert len(train.sentence_lengths) == 1
    assert len(val.sentence_lengths) == 1
    assert len(train) > 0


def test_windows_consistency_guard():
    with pytest.raises(ValueError, match="sentence_lengths"):
        Windows(
            contexts=np.zeros((4, 2), dtype=np.int32),
            targets=np.zeros(4, dtype=np.int32),
            context_size=2,
            sentence_lengths=(3, 3),
        )
