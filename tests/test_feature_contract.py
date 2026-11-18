"""The feature contract: a feature is a class, and the class checks itself.

A malformed feature must fail when the class is created (import time), not when
a user first selects it. That is the whole reason features are classes: the
required attributes and method are declared once and verified automatically.
"""

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.features import Feature
from orangecontrib.ai_suite.kernel.features.catalogue import _build
from orangecontrib.ai_suite.kernel.features.description import Description

GOOD = Description(
    title="Probe",
    summary="A sentence long enough to pass the check.",
    intuition=(
        "A plain-language paragraph long enough to pass the length check, saying which "
        "kinds of picture move the value and which leave it alone."
    ),
    technical="A technical account long enough to pass the check, saying how the value is computed.",
)


def _make(name, description=GOOD, **extra):
    """Build a Feature subclass the way a catalogue module does."""

    namespace = {
        "__module__": __name__,  # type() would take abc.py's frame instead
        "name": name,
        "description": description,
        "compute": lambda self, image, options: 0.0,
    }
    namespace.update(extra)
    return type("Probe", (Feature,), namespace)


def test_valid_feature_class_is_accepted():
    # The class name must match the module it is defined in; this test module
    # is not named after any feature, so declare one that matches.
    feature = _make("test_feature_contract")
    assert feature().compute(np.zeros((2, 2)), None) == 0.0


def test_name_must_be_snake_case():
    for bad in ("HoleCount", "hole-count", "", "hole__count", "1hole"):
        with pytest.raises(TypeError, match="snake_case"):
            _make(bad)


def test_name_must_match_the_module_filename():
    with pytest.raises(TypeError, match="must live in"):
        _make("hole_count")


def test_description_must_be_a_description():
    with pytest.raises(TypeError, match="description must be a Description"):
        _make("test_feature_contract", description="not a Description")


def test_flags_must_be_bools():
    with pytest.raises(TypeError, match="is_count"):
        _make("test_feature_contract", is_count=1)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("title", "no", "title must be"),
        ("title", "A title with a full stop.", "not a sentence"),
        ("title", "lower case", "capital"),
        ("summary", "No full stop", "one capitalised sentence"),
        ("summary", "x" * 141 + ".", "1-140"),
        ("summary", "Line\nbreak.", "single line"),
        ("intuition", "too short", "60-600"),
        ("intuition", "x" * 601, "60-600"),
        ("technical", "too short", "40-1400"),
        ("technical", "x" * 1401, "40-1400"),
    ],
)
def test_description_fields_are_checked(field, value, message):
    with pytest.raises(ValueError, match=message):
        Description(**{**GOOD.__dict__, field: value})


def test_intuition_rejects_technical_vocabulary():
    with pytest.raises(ValueError, match="technical vocabulary"):
        Description(**{**GOOD.__dict__, "intuition": GOOD.intuition + " The variance is large."})


@pytest.mark.parametrize("reference", ["not a link", "[Text](not-a-url)", "[Text]()"])
def test_references_must_be_markdown_links(reference):
    with pytest.raises(ValueError, match="Markdown link"):
        Description(**{**GOOD.__dict__, "references": (reference,)})


def test_catalogue_rejects_duplicate_names():
    first = _make("test_feature_contract")()
    second = _make("test_feature_contract")()
    with pytest.raises(RuntimeError, match="duplicate feature name"):
        _build([first, second])


def test_catalogue_rejects_duplicate_summaries():
    shared = Description(**{**GOOD.__dict__, "title": "Other"})
    first = _make("test_feature_contract")()
    second = _make("test_feature_contract", description=shared)()
    with pytest.raises(RuntimeError, match="repeats the summary"):
        _build([first, second])


def test_catalogue_rejects_duplicate_titles():
    other = Description(**{**GOOD.__dict__, "summary": "A different sentence, long enough."})
    first = _make("test_feature_contract")()
    second = _make("test_feature_contract", description=other)()
    with pytest.raises(RuntimeError, match="repeats the title"):
        _build([first, second])


def test_abstract_compute_cannot_be_instantiated():
    class Incomplete(Feature):
        name = "test_feature_contract"
        description = GOOD

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
