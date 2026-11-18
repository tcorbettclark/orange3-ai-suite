"""The package façade: the names a caller needs, and nothing that is not.

Pinned deliberately. Widening the surface should be a decision that breaks a
test, not something that happens by adding an import.
"""

import orangecontrib.ai_suite.kernel.lm as lm

EXPECTED = frozenset(
    {
        "Architecture",
        "Bigram",
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


def test_package_exports_only_the_use_case():
    assert set(lm.__all__) == EXPECTED
    for name in lm.__all__:
        assert hasattr(lm, name), name


def test_internals_live_in_their_own_module():
    # Supported, but not part of the façade: the special ids are an invariant of
    # the encoding, and deserialisation belongs to whoever owns the format.
    from orangecontrib.ai_suite.kernel.lm.architecture import architecture_from_dict
    from orangecontrib.ai_suite.kernel.lm.tokenizer import BOS_ID, EOS_ID, UNK_ID
    from orangecontrib.ai_suite.kernel.lm.training import EpochStats

    assert (BOS_ID, UNK_ID, EOS_ID) == (0, 1, 2)
    assert callable(architecture_from_dict)
    assert EpochStats is dict
