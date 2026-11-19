"""The package façade: the names a caller needs, and nothing that is not.

``EXPECTED`` is the whole surface, compared against ``lm.__all__``.
"""

import math

import pytest

import edu_ai_suite.lm as lm

EXPECTED = frozenset(
    {
        "Architecture",
        "Bigram",
        "EPOCH_STATS_KEYS",
        "LSTM",
        "LanguageModel",
        "MLP",
        "Tokenizer",
        "TrainingOptions",
        "architectures",
        "make_pairs",
        "tokenizer_from_text",
        "tokenizers",
    }
)


def test_the_epoch_stats_keys_are_a_view_of_the_record_they_name():
    # One spelling: the keys are the record's own annotations, in order, so the
    # checkpoint columns cannot be a list somebody forgot to update.
    from edu_ai_suite.lm.stats import EPOCH_STATS_KEYS, EpochStats

    assert list(EPOCH_STATS_KEYS) == list(EpochStats.__annotations__)


def test_an_epoch_stats_record_puts_each_number_under_its_own_key():
    # The assertion a key-order test cannot make: distinguishable values, read
    # back by name, so a record built in the wrong order fails here.
    from edu_ai_suite.lm.training import epoch_callback

    seen = []
    epoch_callback(on_epoch=seen.append).on_epoch_end(0, {"loss": 0.5, "val_loss": 1.5})
    stats = seen[0]
    assert stats["epoch"] == 1
    assert stats["loss"] == pytest.approx(0.5)
    assert stats["val_loss"] == pytest.approx(1.5)
    assert stats["perplexity"] == pytest.approx(math.exp(0.5))
    assert stats["val_perplexity"] == pytest.approx(math.exp(1.5))


def test_an_epoch_with_no_metric_reports_nan_rather_than_raising():
    from edu_ai_suite.lm.training import epoch_callback

    seen = []
    epoch_callback(on_epoch=seen.append).on_epoch_end(2, {})
    stats = seen[0]
    assert stats["epoch"] == 3
    for name in ("loss", "val_loss", "perplexity", "val_perplexity"):
        assert math.isnan(stats[name]), name


def test_a_hook_may_add_keys_of_its_own():
    # The record is a plain dict because the caller extends it: the per-epoch
    # sample is the hook's addition, and the checkpoint table carries it.
    from edu_ai_suite.lm.training import epoch_callback

    seen = []

    def hook(stats):
        stats["sample"] = "once upon a time"
        seen.append(stats)

    epoch_callback(on_epoch=hook).on_epoch_end(0, {"loss": 0.1})
    assert seen[0]["sample"] == "once upon a time"
    assert seen[0]["loss"] == pytest.approx(0.1)


def test_package_exports_only_the_use_case():
    assert set(lm.__all__) == EXPECTED
    for name in lm.__all__:
        assert hasattr(lm, name), name


def test_internals_live_in_their_own_module():
    # Supported, but not part of the façade: the special ids are an invariant of
    # the encoding, and deserialisation belongs to whoever owns the format.
    from edu_ai_suite.lm.architecture import architecture_from_dict
    from edu_ai_suite.lm.stats import EpochStats
    from edu_ai_suite.lm.tokenizer import BOS_ID, EOS_ID, UNK_ID

    assert (BOS_ID, UNK_ID, EOS_ID) == (0, 1, 2)
    assert callable(architecture_from_dict)
    assert EpochStats(epoch=1)["epoch"] == 1  # a record, built by name
