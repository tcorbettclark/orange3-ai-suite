"""The flat feature table: one entry per feature, in column order.

The table is built from the catalogue roster, and column order *is* roster
order, so there is no second place for an index to go stale.

Nothing here knows about groups. Grouping is a set of membership lists over
these names and lives in :mod:`groups`; this module only answers "which
features are there, and in what order".
"""

from __future__ import annotations

from collections.abc import Iterable

from .catalogue import CATALOGUE
from .feature import Feature


def _build(measures: Iterable[Feature]) -> dict[str, Feature]:
    table: dict[str, Feature] = {}
    definitions: dict[str, str] = {}
    problems: list[str] = []
    for feature in measures:
        if feature.name in table:
            problems.append(f"duplicate feature name {feature.name!r}")
        if feature.definition in definitions:
            problems.append(
                f"{feature.name!r} repeats the definition of {definitions[feature.definition]!r}"
            )
        definitions[feature.definition] = feature.name
        table[feature.name] = feature
    if problems:
        raise RuntimeError("; ".join(problems))
    return table


_BY_NAME: dict[str, Feature] = _build(CATALOGUE)

#: Column position of each feature, derived once from the roster order.
POSITION: dict[str, int] = {name: i for i, name in enumerate(_BY_NAME)}


def all_features() -> tuple[Feature, ...]:
    """All features in stable column order."""
    return tuple(_BY_NAME.values())


def all_feature_names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def all_count_feature_names() -> tuple[str, ...]:
    """Names of the integer-valued count features, in column order."""
    return tuple(feature.name for feature in _BY_NAME.values() if feature.is_count)


def is_known(name: str) -> bool:
    """Whether a feature name exists in the catalogue."""
    return name in _BY_NAME


def get_feature(name: str) -> Feature:
    """Look up one feature by name; a friendly error for unknown names."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown feature {name!r}; see all_feature_names() ({len(_BY_NAME)} available)"
        ) from None
