"""Feature extraction: one image in, a :class:`FeatureVector` out."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from ._preprocessing import preprocess, preprocess_colour
from .catalogue import POSITION, get_feature, is_known
from .groups import by_group, group_of, selection
from .options import FeatureOptions


class FeatureVector(Mapping[str, float]):
    """Result of feature extraction: a read-only name -> value mapping in column order.

    Values are held in catalogue column order whatever order they were supplied in,
    so :meth:`names` and :meth:`to_array` always line up. Feature metadata
    (description, count flag) lives in the catalogue, ``get_feature(name)``, and
    group views are delegated to :mod:`groups`.
    """

    def __init__(self, values: Mapping[str, float]):
        unknown = sorted(name for name in values if not is_known(name))
        if unknown:
            raise ValueError(f"unknown feature names: {unknown}")
        self._values = {
            name: float(values[name]) for name in sorted(values, key=POSITION.__getitem__)
        }

    def __getitem__(self, name: str) -> float:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FeatureVector({len(self._values)} features)"

    def names(self) -> tuple[str, ...]:
        """Feature names in catalogue column order."""
        return tuple(self._values)

    def to_array(self) -> np.ndarray:
        """Values in catalogue column order, as one flat array."""
        return np.asarray(list(self._values.values()), dtype=np.float64)

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
    """Feature values for one image, as a :class:`FeatureVector`.

    Parameters
    ----------
    image:
        RGB(A) or grayscale array in [0, 1] or [0, 255]; channel order is RGB
        (convert OpenCV's BGR before calling).
    options:
        :class:`FeatureOptions`; defaults applied when None.
    groups:
        None (the default groups, see :mod:`groups`) or a subset string such
        as ``"AD"``. The colour shortcut group is opt-in only.
    exclude:
        None, a single feature name, or an iterable of names to drop -
        per-feature switching on top of group selection.
    """
    options = options or FeatureOptions()
    selected = selection(groups)

    excluded = frozenset((exclude,) if isinstance(exclude, str) else exclude or ())
    unknown_features = sorted(name for name in excluded if not is_known(name))
    if unknown_features:
        raise ValueError(f"unknown feature names to exclude: {unknown_features}")

    features = tuple(get_feature(name) for name in selected if name not in excluded)
    if not features:
        raise ValueError("no features selected (all groups excluded or empty)")

    # The pipeline is a property of the feature, not of the group it sits in.
    needs_colour = any(feature.needs_colour for feature in features)
    needs_luminance = any(not feature.needs_colour for feature in features)
    luminance = preprocess(image, options) if needs_luminance else None
    colour = preprocess_colour(image, options) if needs_colour else None

    values = {
        feature.name: feature.compute(colour if feature.needs_colour else luminance, options)
        for feature in features
    }
    return FeatureVector(values)
