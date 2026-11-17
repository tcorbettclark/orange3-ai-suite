"""Interpretable image features, one class per feature.

Every feature is a class in :mod:`catalogue` carrying its own name, definition
and computation, and every value is a human-auditable sentence. Where the
features read their input from is part of the feature (``needs_colour``), not
of any grouping.

Feature groups are membership lists, not a structure: :mod:`groups` holds them
over the flat feature table in :mod:`index`, so the same feature can belong to
several groups and a group is only ever a selection for ablation or display.

Use :func:`all_features` / :func:`get_feature` for the catalogue in column
order, and :func:`all_groups` / :func:`get_group` for the group letters and
their titles.
"""

from .extraction import FeatureVector, extract_features
from .feature import Feature
from .groups import (
    DEFAULT_GROUPS,
    GROUP_ORDER,
    GROUPS,
    Group,
    all_groups,
    check_groups,
    features_of_group,
    get_group,
    group_names,
)
from .index import all_count_feature_names, all_feature_names, all_features, get_feature
from .options import FeatureOptions
from .preprocessing import preprocess, preprocess_colour, to_luminance

__all__ = [
    "DEFAULT_GROUPS",
    "GROUPS",
    "GROUP_ORDER",
    "Feature",
    "FeatureOptions",
    "FeatureVector",
    "Group",
    "all_count_feature_names",
    "all_feature_names",
    "all_features",
    "all_groups",
    "check_groups",
    "extract_features",
    "features_of_group",
    "get_feature",
    "get_group",
    "group_names",
    "preprocess",
    "preprocess_colour",
    "to_luminance",
]
