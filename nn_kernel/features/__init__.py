"""Image feature extraction (groups A-F).

Domain doc: spec/image-feature-extractor-widget.md. Every feature is a
human-auditable sentence — the per-feature definition sentences are the
canonical documentation and live in the registry (built from each group
module's ``DEFINITIONS``). See registry.all_features() / get_feature().
"""

from .options import FeatureOptions
from .preprocess import preprocess, preprocess_colour, to_luminance
from .registry import (
    DEFAULT_GROUPS,
    GROUP_ORDER,
    FEATURES,
    Feature,
    all_features,
    all_feature_names,
    features_of_group,
    get_feature,
    group_names,
)
from .extract import FeatureVector, extract_features

__all__ = [
    "FeatureOptions",
    "FeatureVector",
    "extract_features",
    "preprocess",
    "preprocess_colour",
    "to_luminance",
    "Feature",
    "FEATURES",
    "DEFAULT_GROUPS",
    "GROUP_ORDER",
    "all_features",
    "all_feature_names",
    "features_of_group",
    "get_feature",
    "group_names",
]