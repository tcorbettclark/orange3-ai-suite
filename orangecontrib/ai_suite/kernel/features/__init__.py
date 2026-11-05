"""Interpretable image features (groups A-G).

Every feature is a human-auditable sentence. The definition sentences live in
the group modules (``DEFINITIONS``) and are collected by :mod:`registry`, which
is the single index and the canonical documentation:

- groups A-F are computed on a preprocessed luminance image;
- group G reads colour directly and is opt-in - it demonstrates shortcut
  learning (``bluish_fraction`` alone can solve "is there sky?").

Use :func:`all_features` / :func:`get_feature` to list or look up features.
"""

from .extract import FeatureVector, extract_features
from .options import FeatureOptions
from .preprocess import preprocess, preprocess_colour, to_luminance
from .registry import (
    DEFAULT_GROUPS,
    FEATURES,
    GROUP_ORDER,
    Feature,
    all_feature_names,
    all_features,
    features_of_group,
    get_feature,
    group_names,
)

__all__ = [
    "DEFAULT_GROUPS",
    "FEATURES",
    "GROUP_ORDER",
    "Feature",
    "FeatureOptions",
    "FeatureVector",
    "all_feature_names",
    "all_features",
    "extract_features",
    "features_of_group",
    "get_feature",
    "group_names",
    "preprocess",
    "preprocess_colour",
    "to_luminance",
]
