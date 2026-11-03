"""The single index of image features.

Each feature is a first-class :class:`Feature` carrying its name, group letter,
stable column order and human-auditable definition sentence. One module per
group provides the computation (``compute``) and the declarative metadata
(``FEATURE_NAMES``, ``DEFINITIONS``); the registry stitches them together and
validates consistency at import time.

Selection happens at group level (``extract_features(..., groups="AD")``) or at
individual-feature level (``exclude={...}``).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import colour, photometry, profiles, spatial, symmetry, texture, topology

GROUP_ORDER = ("A", "B", "C", "D", "E", "F", "G")
DEFAULT_GROUPS = "ABCDEF"  # the colour shortcut group is opt-in

_MODULES = {
    "A": photometry,
    "B": profiles,
    "C": spatial,
    "D": texture,
    "E": topology,
    "F": symmetry,
    "G": colour,
}


@dataclass(frozen=True)
class Feature:
    """One image feature: identity, group, column order and definition."""

    name: str
    group: str  # "A".."G"
    definition: str  # the human-auditable sentence
    index: int  # stable column order across the whole suite


def _build() -> dict[str, Feature]:
    out: dict[str, Feature] = {}
    for letter in GROUP_ORDER:
        module = _MODULES[letter]
        names = module.FEATURE_NAMES
        definitions = module.DEFINITIONS
        problems = []
        if len(set(names)) != len(names):
            problems.append(f"duplicate feature names in group {letter}")
        missing = set(names) - set(definitions)
        extra = set(definitions) - set(names)
        if missing:
            problems.append(f"group {letter}: features without a definition: {sorted(missing)}")
        if extra:
            problems.append(f"group {letter}: definitions without a feature: {sorted(extra)}")
        if problems:
            raise RuntimeError("; ".join(problems))
        for name in names:
            out[name] = Feature(name, letter, definitions[name], len(out))
    return out


FEATURES: dict[str, Feature] = _build()


def all_features() -> tuple[Feature, ...]:
    """All features in stable column order."""
    return tuple(FEATURES.values())


def all_feature_names() -> tuple[str, ...]:
    return tuple(feature.name for feature in all_features())


def features_of_group(group: str) -> tuple[Feature, ...]:
    """Features of one group letter, in column order."""
    if group not in GROUP_ORDER:
        raise ValueError(f"unknown feature group {group!r}; valid: {GROUP_ORDER}")
    return tuple(f for f in FEATURES.values() if f.group == group)


def group_names(group: str) -> tuple[str, ...]:
    """Feature names contributed by one group letter."""
    return tuple(f.name for f in features_of_group(group))


def get_feature(name: str) -> Feature:
    """Look up one feature by name; a friendly error for unknown names."""
    try:
        return FEATURES[name]
    except KeyError:
        raise ValueError(
            f"unknown feature {name!r}; see all_feature_names() ({len(FEATURES)} available)"
        ) from None
