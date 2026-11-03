"""Feature extraction front door: FeatureVector and extract_features."""

from __future__ import annotations

import numpy as np

from .options import FeatureOptions
from .preprocess import preprocess
from .registry import GROUP_ORDER, FEATURES, _MODULES


class FeatureVector:
    """Result of feature extraction: ordered name -> value mapping.

    Behaves like a read-only dict (``vec["hole_count"]``, ``vec.items()``)
    and can produce the plain float array the Orange adapter will wrap in a
    Table column by column. Feature metadata (group, definition sentence)
    lives in the registry: ``registry.get_feature(name)``.
    """

    def __init__(self, values: dict[str, float], group_of: dict[str, str]):
        self._values = dict(values)
        self._group_of = dict(group_of)

    def __getitem__(self, name: str) -> float:
        return self._values[name]

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FeatureVector({len(self._values)} features)"

    def names(self) -> tuple[str, ...]:
        return tuple(self._values)

    def items(self):
        return self._values.items()

    def to_array(self) -> np.ndarray:
        return np.asarray([self._values[n] for n in self._values], dtype=np.float64)

    def group_of(self, name: str) -> str:
        """Group letter ('A'..'F') the named feature belongs to."""
        return self._group_of[name]

    def by_group(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, value in self._values.items():
            out.setdefault(self._group_of[name], {})[name] = value
        return out


def extract_features(
    image: np.ndarray,
    options: FeatureOptions | None = None,
    groups: str | None = None,
    exclude: str | None = None,
) -> FeatureVector:
    """Extract features from one image.

    Parameters
    ----------
    image:
        RGB(A) or grayscale array; channel order is RGB (convert cv2's BGR
        in the widget adapter before calling).
    options:
        FeatureOptions; defaults applied when None.
    groups:
        None (all groups) or a subset string such as ``"AD"`` for group
        ablation experiments (domain doc §5).
    exclude:
        None, or an iterable of feature names to drop — per-feature
        switching on top of group selection.
    """
    opts = options or FeatureOptions()
    selected = GROUP_ORDER if groups is None else tuple(groups)
    unknown_groups = set(selected) - set(GROUP_ORDER)
    if unknown_groups:
        raise ValueError(
            f"unknown feature groups {sorted(unknown_groups)}; valid: {GROUP_ORDER}"
        )

    excluded = frozenset(exclude or ())
    unknown_features = excluded - set(FEATURES)
    if unknown_features:
        raise ValueError(f"unknown feature names to exclude: {sorted(unknown_features)}")

    preprocessed = preprocess(image, opts)
    values: dict[str, float] = {}
    group_of: dict[str, str] = {}
    for letter in selected:
        module = _MODULES[letter]
        for name, value in module.compute(preprocessed, opts).items():
            if name in excluded:
                continue
            values[name] = float(value)
            group_of[name] = letter
    if not values:
        raise ValueError("no features selected (all groups excluded or empty)")
    return FeatureVector(values, group_of)