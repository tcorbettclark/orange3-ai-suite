"""The stride both validation splits are built on."""

import pytest

from edu_ai_suite._holdout import held_out_positions


@pytest.mark.parametrize(
    ("n", "fraction", "expected"),
    [
        # step = max(1, round(1 / fraction)), then every step-th position
        (10, 0.1, [0]),
        (10, 0.25, [0, 4, 8]),
        (10, 0.5, [0, 2, 4, 6, 8]),
        (20, 0.1, [0, 10]),
        (4, 0.5, [0, 2]),
        (5, 0.25, [0, 4]),
        # more held out than training: clamped to n - 1, so training keeps a row
        (2, 0.99, [0]),
        (5, 0.99, [0, 1, 2, 3]),
        # fewer items than the stride still holds out exactly one
        (3, 0.1, [0]),
    ],
)
def test_a_stride_over_the_positions(n, fraction, expected):
    assert held_out_positions(n, fraction) == expected


def test_an_offset_past_the_end_falls_back_to_the_last_position():
    # step 10 over 3 items with seed 7: no position is on the stride, and a split
    # with nothing held out is not a split, so the last item is taken.
    assert held_out_positions(3, 0.1, 7) == [2]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(0, [0, 4, 8]), (1, [1, 5, 9]), (2, [2, 6]), (3, [3, 7]), (4, [0, 4, 8])],
)
def test_the_seed_only_shifts_the_stride(seed, expected):
    # 10 items at a quarter: step 4, so there are four possible splits, and the
    # seed picks one - it does not reshuffle anything.
    assert held_out_positions(10, 0.25, seed) == expected


def test_no_split_at_all_below_two_items():
    assert held_out_positions(0, 0.25) == []
    assert held_out_positions(1, 0.25) == []


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_the_fraction_must_be_a_real_fraction(fraction):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        held_out_positions(10, fraction)


@pytest.mark.parametrize("n", range(2, 41))
def test_both_sides_are_never_empty(n):
    for fraction in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        for seed in (0, 1, 7, 99):
            held = held_out_positions(n, fraction, seed)
            assert 1 <= len(held) <= n - 1, (n, fraction, seed, held)
            assert len(set(held)) == len(held)
            assert all(0 <= position < n for position in held)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_the_fraction_is_approximated_within_one_item(fraction):
    # The stride is a rate, not a count: the held-out share tracks the fraction
    # to within a single position on a corpus of a teaching size.
    for n in (10, 40, 400):
        assert abs(len(held_out_positions(n, fraction)) - n * fraction) <= 1
