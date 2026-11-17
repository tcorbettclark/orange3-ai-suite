"""The feature contract: a feature is a class, and the class checks itself.

A malformed feature must fail when the class is created (import time), not when
a user first selects it. That is the whole reason features are classes: the
required attributes and method are declared once and verified automatically.
"""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.features import Feature
from orangecontrib.ai_suite.kernel.features.catalogue import _build


def _make(name, definition, **extra):
    """Build a Feature subclass the way a catalogue module does."""

    namespace = {
        "__module__": __name__,  # type() would take abc.py's frame instead
        "name": name,
        "definition": definition,
        "compute": lambda self, img, opts: 0.0,
    }
    namespace.update(extra)
    return type("Probe", (Feature,), namespace)


def test_valid_feature_class_is_accepted():
    # The class name must match the module it is defined in; this test module
    # is not named after any feature, so declare one that matches.
    feature = _make("test_feature_contract", "A sentence long enough to pass the check.")
    assert feature().compute(np.zeros((2, 2)), None) == 0.0


def test_name_must_be_snake_case():
    for bad in ("HoleCount", "hole-count", "", "hole__count", "1hole"):
        with pytest.raises(TypeError, match="snake_case"):
            _make(bad, "A sentence long enough to pass the check.")


def test_name_must_match_the_module_filename():
    with pytest.raises(TypeError, match="must live in"):
        _make("hole_count", "A sentence long enough to pass the check.")


@pytest.mark.parametrize(
    "bad",
    [
        "lower case start",
        "No trailing period",
        "Short.",
        "A sentence long enough but without a full stop",
    ],
)
def test_definition_must_be_a_sentence(bad):
    with pytest.raises(TypeError, match="definition"):
        _make("test_feature_contract", bad)


def test_flags_must_be_bools():
    with pytest.raises(TypeError, match="is_count"):
        _make(
            "test_feature_contract",
            "A sentence long enough to pass the check.",
            is_count=1,
        )


def test_catalogue_rejects_duplicate_names():
    first = _make("test_feature_contract", "A sentence long enough to pass the check.")()
    second = _make("test_feature_contract", "A different sentence, also long enough.")()
    with pytest.raises(RuntimeError, match="duplicate feature name"):
        _build([first, second])


def test_catalogue_rejects_duplicate_definitions():
    first = _make("test_feature_contract", "One shared sentence, long enough to count.")()
    second = _make("test_feature_contract", "One shared sentence, long enough to count.")()
    with pytest.raises(RuntimeError, match="repeats the definition"):
        _build([first, second])


def test_abstract_compute_cannot_be_instantiated():
    class Incomplete(Feature):
        name = "test_feature_contract"
        definition = "A sentence long enough to pass the check."

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
