"""Feature extraction front door: :class:`FeatureVector` and
:func:`extract_features`."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .groups import by_group, group_of, selection
from .index import POSITION, get_feature, is_known
from .options import FeatureOptions
from .preprocessing import preprocess, preprocess_colour


class FeatureVector:
    """Result of feature extraction: an ordered name -> value mapping.

    Behaves like a read-only dict (``vec["hole_count"]``, ``vec.keys()``,
    ``vec.items()``) and can produce the plain float array used for one Table
    row. Feature metadata (group, definition sentence) lives in the index:
    ``get_feature(name)``.
    """

    def __init__(self, values: dict[str, float]):
        self._values = dict(values)

    def __getitem__(self, name: str) -> float:
        return self._values[name]

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FeatureVector({len(self._values)} features)"

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def get(self, name: str, default: float | None = None) -> float | None:
        return self._values.get(name, default)

    def names(self) -> tuple[str, ...]:
        return tuple(self._values)

    def to_array(self) -> np.ndarray:
        """Values in stable column order."""
        ordered = sorted(self._values, key=POSITION.__getitem__)
        return np.asarray([self._values[name] for name in ordered], dtype=np.float64)

    def group_of(self, name: str) -> str:
        """Group letter the named feature is reported under."""
        return group_of(name)

    def by_group(self, letter: str | None = None) -> dict[str, dict[str, float]]:
        """Values split by group; see :func:`..groups.by_group`."""
        return by_group(self._values, letter)


def extract_features(
    image: np.ndarray,
    options: FeatureOptions | None = None,
    groups: str | None = None,
    exclude: Iterable[str] | None = None,
) -> FeatureVector:
    """Extract features from one image.

    Parameters
    ----------
    image:
        RGB(A) or grayscale array; channel order is RGB (convert OpenCV's BGR
        before calling).
    options:
        :class:`FeatureOptions`; defaults applied when None.
    groups:
        None (all default groups, ``"ABCDEF"``) or a subset string such as
        ``"AD"``. The colour shortcut group G is opt-in only.
    exclude:
        None, or an iterable of feature names to drop - per-feature switching
        on top of group selection.
    """
    opts = options or FeatureOptions()
    selected = selection(groups)

    excluded = frozenset(exclude or ())
    unknown_features = sorted(name for name in excluded if not is_known(name))
    if unknown_features:
        raise ValueError(f"unknown feature names to exclude: {unknown_features}")

    features = tuple(get_feature(name) for name in selected if name not in excluded)
    if not features:
        raise ValueError("no features selected (all groups excluded or empty)")

    # The pipeline is a property of the feature, not of the group it sits in.
    needs_colour = any(feature.needs_colour for feature in features)
    needs_luminance = any(not feature.needs_colour for feature in features)
    luminance = preprocess(image, opts) if needs_luminance else None
    colour = preprocess_colour(image, opts) if needs_colour else None

    values = {
        feature.name: feature.compute(colour if feature.needs_colour else luminance, opts)
        for feature in features
    }
    return FeatureVector(values)
