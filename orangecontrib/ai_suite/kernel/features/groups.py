"""Feature groups: a lightweight assembly over the flat feature table.

A group owns no feature and computes nothing. It is a membership list plus the
words to describe it, so the same feature may appear in several groups - that
overlap is what makes alternative ablations possible without touching the
features themselves.

Two rules keep the assembly honest, both checked at import:

- every name listed by a group exists in the index, and every feature belongs
  to at least one group (an unlisted feature could never be selected);
- a group is either on by default or opt-in, and opt-in groups are the ones
  holding colour features - the shortcut stays a deliberate choice.

Display rule: a feature is reported under the *first* group that lists it, so
:func:`group_of` and the widget tree show each feature exactly once even when
memberships overlap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .feature import Feature
from .index import all_feature_names, get_feature


@dataclass(frozen=True)
class Assembly:
    """One group of features: a letter, the words, and its members."""

    letter: str
    title: str  # e.g. "global photometry"
    description: str  # one paragraph of help text
    members: tuple[str, ...]  # feature names, in column order


ASSEMBLIES: tuple[Assembly, ...] = (
    Assembly(
        "A",
        "global photometry",
        "Global photometry: brightness, contrast, skewness, kurtosis and "
        "bright fraction. Little spatial information; useful as a baseline.",
        (
            "mean_intensity",
            "intensity_std",
            "robust_contrast",
            "skewness",
            "kurtosis",
            "bright_fraction",
        ),
    ),
    Assembly(
        "B",
        "profile summaries",
        "Profile summaries: dense row, column and diagonal projections. "
        "Trend, roughness, peak counts and median crossings.",
        (
            "row_profile_trend",
            "row_profile_roughness",
            "row_profile_peaks",
            "column_profile_trend",
            "column_profile_roughness",
            "column_profile_peaks",
            "diagonal_profile_peaks",
            "anti_diagonal_profile_peaks",
            "profile_row_column_correlation",
            "profile_median_crossing_count",
        ),
    ),
    Assembly(
        "C",
        "spatial mass",
        "Spatial mass: where the bright matter sits, how wide or tall it is, "
        "elongation and the radial gradient.",
        (
            "centre_of_mass_x",
            "centre_of_mass_y",
            "horizontal_spread",
            "vertical_spread",
            "elongation",
            "radial_gradient",
        ),
    ),
    Assembly(
        "D",
        "texture",
        "Texture: gradient strength, edge density, fine and coarse texture, "
        "and autocorrelation width. Carries most signal for cloud imagery.",
        (
            "mean_gradient_magnitude",
            "edge_density",
            "fine_texture",
            "coarse_texture",
            "coarse_to_fine_texture_ratio",
            "autocorrelation_half_width",
        ),
    ),
    Assembly(
        "E",
        "topology",
        "Topology: Otsu foreground, enclosed-background hole count and stroke "
        "crossings. The classic digit features.",
        ("hole_count", "foreground_area_fraction", "stroke_crossings_median_level"),
    ),
    Assembly(
        "F",
        "symmetry",
        "Symmetry: left-right, top-bottom and quadrant self-similarity.",
        ("left_right_symmetry", "top_bottom_symmetry", "quadrant_self_similarity"),
    ),
    Assembly(
        "G",
        "colour",
        "Colour: bluish fraction, blueness, saturation and greenery. A "
        "deliberate shortcut, off by default: colour alone can answer 'is "
        "there sky?' without learning any spatial structure.",
        ("bluish_fraction", "mean_blueness", "mean_saturation", "greenery_fraction"),
    ),
)

GROUP_ORDER: tuple[str, ...] = tuple(assembly.letter for assembly in ASSEMBLIES)
DEFAULT_GROUPS = "ABCDEF"  # the colour shortcut group is opt-in


def _validate() -> dict[str, tuple[str, ...]]:
    """Check the assembly against the index; returns name -> member letters."""
    problems: list[str] = []
    memberships: dict[str, list[str]] = {}
    if len(set(GROUP_ORDER)) != len(GROUP_ORDER):
        problems.append("duplicate group letters")
    for assembly in ASSEMBLIES:
        if not assembly.title or not assembly.description:
            problems.append(f"group {assembly.letter}: missing title or description")
        if len(set(assembly.members)) != len(assembly.members):
            problems.append(f"group {assembly.letter}: duplicate members")
        for name in assembly.members:
            get_feature(name)  # raises with a friendly message
            memberships.setdefault(name, []).append(assembly.letter)

    unlisted = set(all_feature_names()) - set(memberships)
    if unlisted:
        problems.append(f"features in no group: {sorted(unlisted)}")
    for letter in DEFAULT_GROUPS:
        if letter not in GROUP_ORDER:
            problems.append(f"DEFAULT_GROUPS names unknown group {letter!r}")

    default_members = {
        name
        for assembly in ASSEMBLIES
        if assembly.letter in DEFAULT_GROUPS
        for name in assembly.members
    }
    colour = sorted(name for name in default_members if get_feature(name).needs_colour)
    if colour:
        problems.append(f"colour features must be opt-in: {colour} is on by default")
    if problems:
        raise RuntimeError("; ".join(problems))
    return {name: tuple(letters) for name, letters in memberships.items()}


_MEMBERSHIPS: dict[str, tuple[str, ...]] = _validate()


def all_groups() -> tuple[Assembly, ...]:
    """All groups, in display order."""
    return ASSEMBLIES


def get_group(letter: str) -> Assembly:
    """Look up one group by letter."""
    check_groups((letter,))
    for assembly in ASSEMBLIES:
        if assembly.letter == letter:
            return assembly
    raise AssertionError(f"group {letter!r} is in GROUP_ORDER but has no assembly")


def check_groups(letters: Iterable[str]) -> tuple[str, ...]:
    """Validate a group selection and return it as a tuple of letters."""
    selection = tuple(letters)
    unknown = sorted(set(selection) - set(GROUP_ORDER))
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; valid: {GROUP_ORDER}")
    return selection


def memberships_of(name: str) -> tuple[str, ...]:
    """Group letters that list this feature, in display order."""
    get_feature(name)
    return _MEMBERSHIPS[name]


def group_of(name: str) -> str:
    """The group a feature is reported under (its first membership)."""
    return memberships_of(name)[0]


def group_names(letter: str) -> tuple[str, ...]:
    """Feature names contributed by one group, in column order."""
    return get_group(letter).members


def features_of_group(letter: str) -> tuple[Feature, ...]:
    """Features of one group letter, in column order."""
    return tuple(get_feature(name) for name in group_names(letter))


def selection(groups: str | None = None) -> tuple[str, ...]:
    """Resolve a group selection to feature names, in stable column order.

    ``None`` means the default selection. A selection may name any subset of
    groups; members are de-duplicated, so overlapping groups are harmless.
    """
    letters = check_groups(DEFAULT_GROUPS if groups is None else groups)
    wanted = {name for letter in letters for name in group_names(letter)}
    return tuple(name for name in all_feature_names() if name in wanted)


def by_group(values: Mapping[str, float], letter: str | None = None) -> dict[str, dict[str, float]]:
    """Split ``name -> value`` by group membership.

    With a ``letter``, only that group's members are kept. Without one, each
    feature is reported under the first group that lists it.
    """
    if letter is not None:
        check_groups((letter,))
    out: dict[str, dict[str, float]] = {}
    for name, value in values.items():
        if letter is None:
            key = group_of(name)
        elif letter in memberships_of(name):
            key = letter
        else:
            continue
        out.setdefault(key, {})[name] = value
    return out
