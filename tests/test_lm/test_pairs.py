"""Pair-building tests - BOS padding and unit-boundary discipline.

``make_pairs`` takes an EOS-terminated token stream, so every case is tested
on hand-written integers: no vocabulary, no corpus, no tokenizer.

A unit of ids ``2, 3, 4`` is written ``[2, 3, 4, EOS]`` in the stream, and it
contributes four pairs: three for its tokens and one for its EOS. Targets are
therefore every token in the stream, in order - ``len(stream)`` pairs, always.
"""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.lm import make_pairs, tokenizer_from_text
from orangecontrib.ai_suite.kernel.lm.pairs import Pairs

BOS, EOS = 0, 2
A, B, C, D, E, F = 3, 4, 5, 6, 7, 8  # ids 0-2 are specials


# ---- pair arithmetic -------------------------------------------------------


def test_bos_padding_and_eos_target():
    # [A, B, C, EOS] with w=3:
    # [B,B,B]->A, [B,B,A]->B, [B,A,B]->C, [A,B,C]->EOS
    pairs = make_pairs([A, B, C, EOS], context_width=3)
    assert np.array_equal(
        pairs.contexts,
        np.array([[0, 0, 0], [0, 0, A], [0, A, B], [A, B, C]], dtype=np.int32),
    )
    assert np.array_equal(pairs.targets, np.array([A, B, C, EOS], dtype=np.int32))
    assert pairs.pairs_per_unit == (4,)  # 3 tokens + 1 EOS target
    assert pairs.context_width == 3


def test_pairs_never_cross_units():
    pairs = make_pairs([A, B, C, EOS, D, E, F, EOS], context_width=3)
    assert len(pairs) == 8
    assert pairs.pairs_per_unit == (4, 4)
    # no id from one unit appears in the other's pairs
    assert not set(pairs.contexts[4:].ravel()) & {A, B, C}
    assert not set(pairs.contexts[:4].ravel()) & {D, E, F}
    # the second unit opens with all-BOS context, not a tail of the first
    assert np.array_equal(pairs.contexts[4], [BOS, BOS, BOS])
    assert np.array_equal(pairs.targets[4:], np.array([D, E, F, EOS], dtype=np.int32))


def test_context_slides_within_a_unit():
    pairs = make_pairs([A, B, C, D, E, EOS], context_width=2)
    # [B B]->A, [B A]->B, [A B]->C, [B C]->D, [C D]->E, [D E]->EOS
    assert np.array_equal(
        pairs.contexts,
        np.array([[0, 0], [0, A], [A, B], [B, C], [C, D], [D, E]], dtype=np.int32),
    )
    assert np.array_equal(pairs.targets, np.array([A, B, C, D, E, EOS], dtype=np.int32))


def test_large_context_all_bos_fill():
    pairs = make_pairs([A, B, C, EOS], context_width=10)
    assert pairs.contexts.shape == (4, 10)
    # the third target (C): 7 BOS pads + [BOS, A, B]
    assert np.array_equal(pairs.contexts[2, :7], np.zeros(7, dtype=np.int32))
    assert np.array_equal(pairs.contexts[2, 7:], np.array([0, A, B], dtype=np.int32))
    # the EOS target: 6 pads + the whole unit behind BOS
    assert np.array_equal(pairs.contexts[3, :6], np.zeros(6, dtype=np.int32))
    assert np.array_equal(pairs.contexts[3, 6:], np.array([0, A, B, C], dtype=np.int32))
    assert pairs.targets[3] == EOS


def test_context_width_one():
    pairs = make_pairs([A, B, C, EOS], context_width=1)
    assert np.array_equal(pairs.contexts.ravel(), [0, A, B, C])
    assert np.array_equal(pairs.targets, [A, B, C, EOS])


def test_single_token_units():
    pairs = make_pairs([A, EOS, B, EOS], context_width=2)
    assert pairs.pairs_per_unit == (2, 2)
    # each unit: [B,B]->token, [B,token]->EOS
    assert np.array_equal(pairs.contexts, [[0, 0], [0, A], [0, 0], [0, B]])
    assert np.array_equal(pairs.targets, [A, EOS, B, EOS])


def test_bos_is_never_a_target_but_eos_is():
    pairs = make_pairs([A, B, EOS, C, EOS], context_width=2)
    assert BOS not in pairs.targets  # BOS is context construction, not data
    assert list(pairs.targets[[2, 4]]) == [EOS, EOS]  # every unit ends on EOS


def test_error_paths():
    with pytest.raises(ValueError, match="context_width"):
        make_pairs([A, EOS], context_width=0)
    with pytest.raises(ValueError, match="must end with <eos>"):
        make_pairs([], context_width=2)
    with pytest.raises(ValueError, match="must end with <eos>"):
        make_pairs([A, B, C], context_width=2)  # unterminated stream
    with pytest.raises(ValueError, match="empty unit"):
        make_pairs([A, EOS, EOS], context_width=2)
    with pytest.raises(ValueError, match="empty unit"):
        make_pairs([EOS, A, EOS], context_width=2)


# ---- end to end: an encoded corpus is a self-describing stream -------------


def test_encoded_corpus_needs_no_text():
    tok = tokenizer_from_text("AB.\nCD.", "char")
    stream = tok.encode("AB.\nCD.")
    assert stream == tok.encode("AB.") + tok.encode("CD.")
    pairs = make_pairs(stream, context_width=3)
    assert pairs.pairs_per_unit == (4, 4)
    assert len(pairs) == len(stream)
    assert np.array_equal(pairs.contexts[4], [BOS, BOS, BOS])


def test_documents_are_not_merged_into_one_unit():
    # "AB" has no terminator: a space-joined corpus would fuse the two documents,
    # a newline-joined one cannot - the same rule that splits inside a document.
    tok = tokenizer_from_text("AB\nCD", "char")
    pairs = make_pairs(tok.encode("AB\nCD"), context_width=1)
    assert pairs.pairs_per_unit == (3, 3)


# ---- splitting -------------------------------------------------------------


def twelve_units() -> Pairs:
    stream = [x for i in range(12) for x in (i + 3, i + 4, EOS)]
    return make_pairs(stream, context_width=2)


def test_split_is_unit_disjoint_deterministic_and_complete():
    pairs = twelve_units()
    train, val = pairs.split(validation_fraction=0.25, seed=1)
    assert len(val.pairs_per_unit) == max(1, round(12 * 0.25)) == 3
    assert sum(train.pairs_per_unit) == len(train)
    assert sum(val.pairs_per_unit) == len(val)
    assert len(train) + len(val) == len(pairs)

    # complete partition: as multisets, (context, target) rows are preserved
    def multiset(w):
        pairs = [tuple(r) + (int(t),) for r, t in zip(w.contexts, w.targets, strict=True)]
        return sorted(pairs)

    assert multiset(pairs) == sorted(multiset(train) + multiset(val))

    # deterministic under a fixed seed
    t2, v2 = pairs.split(validation_fraction=0.25, seed=1)
    assert np.array_equal(train.targets, t2.targets)
    assert np.array_equal(val.contexts, v2.contexts)


def test_split_fraction_validation():
    pairs = make_pairs([A, B, EOS, C, D, EOS], context_width=1)
    with pytest.raises(ValueError, match="validation_fraction"):
        pairs.split(validation_fraction=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        make_pairs([A, B, EOS], context_width=1).split()


def test_split_caps_validation_to_leave_training_rows():
    pairs = make_pairs([A, B, EOS, C, D, EOS], context_width=1)
    train, val = pairs.split(validation_fraction=0.99)
    assert len(train.pairs_per_unit) == 1
    assert len(val.pairs_per_unit) == 1
    assert len(train) > 0


# ---- structural guards: unit structure lives in the targets ---------------


def test_targets_must_end_with_eos():
    with pytest.raises(ValueError, match="must end with <eos>"):
        Pairs(
            contexts=np.zeros((2, 2), dtype=np.int32),
            targets=np.array([3, 4], dtype=np.int32),  # unterminated
            context_width=2,
        )


def test_contexts_shape_is_checked():
    with pytest.raises(ValueError, match="shaped"):
        Pairs(
            contexts=np.zeros((2, 3), dtype=np.int32),
            targets=np.array([3, EOS], dtype=np.int32),
            context_width=2,
        )


def test_pairs_per_unit_is_derived_from_targets():
    # Nothing is stored: the EOS targets are the only record of unit structure.
    pairs = Pairs(
        contexts=np.zeros((5, 1), dtype=np.int32),
        targets=np.array([3, EOS, 4, 5, EOS], dtype=np.int32),
        context_width=1,
    )
    assert pairs.pairs_per_unit == (2, 3)
    assert sum(pairs.pairs_per_unit) == len(pairs)
    assert "pairs_per_unit" not in vars(pairs)


# ---- split keying: content, not corpus order ------------------------------


def _val_first_targets(stream, seed=0, validation_fraction=0.25):
    pairs = make_pairs(stream, 1)
    _, val = pairs.split(validation_fraction=validation_fraction, seed=seed)
    starts = [0] + [i + 1 for i, t in enumerate(val.targets[:-1]) if t == EOS]
    return [int(val.targets[s]) for s in starts]


def test_split_is_independent_of_corpus_order():
    # Same units delivered in a different order: the same units are held out.
    units = [[3, 4], [5, 6], [7, 8], [9, 10]]
    stream = [x for unit in units for x in (*unit, EOS)]
    reversed_stream = [x for unit in reversed(units) for x in (*unit, EOS)]
    assert _val_first_targets(stream) == _val_first_targets(reversed_stream)


def test_identical_units_are_never_split():
    # A duplicated unit hashes alike, so both copies land on the same side.
    stream = [x for u in ([3, 4], [5, 6], [3, 4], [7, 8]) for x in (*u, EOS)]
    for seed in range(20):
        pairs = make_pairs(stream, 1)
        _train, val = pairs.split(validation_fraction=0.5, seed=seed)
        in_val = sum(1 for t in val.targets[::3] if int(t) == 3)
        assert in_val in (0, 2), f"seed {seed} split the duplicate: {in_val} in validation"


def test_split_honours_the_ratio_when_every_unit_is_identical():
    # Keeping duplicates apart is impossible here, and a one-unit validation set
    # would be useless, so the ratio wins and training still keeps a unit.
    stream = [x for _ in range(3) for x in (3, 4, EOS)]
    train, val = make_pairs(stream, 1).split(validation_fraction=0.99)
    assert len(val.pairs_per_unit) == 2
    assert len(train.pairs_per_unit) == 1


def test_split_seed_changes_the_assignment():
    stream = [x for i in range(12) for x in (i + 3, i + 4, EOS)]
    pairs = make_pairs(stream, 2)
    _, first = pairs.split(validation_fraction=0.25, seed=0)
    _, second = pairs.split(validation_fraction=0.25, seed=1)
    assert not np.array_equal(first.targets, second.targets)
