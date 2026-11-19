"""Interpretable image features, one class per feature.

:mod:`catalogue` holds the feature classes, :mod:`groups` the membership lists
that select them, :mod:`extraction` the pipeline that computes them for one
image, and :mod:`guide` the Markdown page that describes them; a leading
underscore marks support modules.

:func:`extract_features` gives one image's values, :func:`all_features` and
:func:`get_feature` the catalogue in column order, :func:`selection` a group's
features, and :func:`feature_guide_markdown` the guide as one page.
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
