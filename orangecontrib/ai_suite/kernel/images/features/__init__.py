"""Interpretable image features, one class per feature.

Every feature is a class in :mod:`catalogue` carrying its own name, description
and computation, and every value is a human-auditable sentence. Where the
features read their input from is part of the feature (``needs_colour``), not
of any grouping.

Feature groups are membership lists, not a structure: :mod:`groups` holds them
over the catalogue, so the same feature can belong to several groups and a
group is only ever a selection for ablation or display.

Use :func:`feature_guide_markdown` for the whole guide as one page,
:func:`extract_features` for values, :func:`all_features` /
:func:`get_feature` for the catalogue in column order, :func:`all_groups` /
:func:`get_group` for the group letters and their titles, and
:func:`selection` to resolve a group string to feature names.

Preprocessing (:mod:`_preprocessing`) is internal: a feature reads its image
through :func:`extract_features`, which applies the one pipeline the whole
catalogue is defined against.

A leading underscore marks support code: modules that only this package's own
modules import (``_conventions``, ``_measurements``, ``_preprocessing``). The
other modules are content or front door - ``catalogue``, ``groups``,
``extraction``, ``feature``, ``description``, ``options``, ``guide`` - and may
be imported directly.
"""

from .catalogue import all_count_feature_names, all_feature_names, all_features, get_feature
from .description import Description
from .extraction import FeatureVector, extract_features
from .feature import Feature
from .groups import (
    DEFAULT_GROUPS,
    GROUP_ORDER,
    Group,
    all_groups,
    features_of_group,
    get_group,
    group_of,
    selection,
)
from .guide import feature_guide_markdown
from .options import FeatureOptions

__all__ = [
    "DEFAULT_GROUPS",
    "GROUP_ORDER",
    "Description",
    "Feature",
    "FeatureOptions",
    "FeatureVector",
    "Group",
    "all_count_feature_names",
    "all_feature_names",
    "all_features",
    "all_groups",
    "extract_features",
    "feature_guide_markdown",
    "features_of_group",
    "get_feature",
    "get_group",
    "group_of",
    "selection",
]
