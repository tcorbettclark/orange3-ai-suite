"""The feature contract: one class, one number, one definition.

A feature is a class, not a module, so its required surface is explicit and
checked when the class is created. Subclasses declare ``name``, ``definition``
and optionally ``is_count`` / ``needs_colour``, then implement :meth:`compute`.

The checks below run at import time, which is deliberate: a malformed feature
should fail when the package loads, not when a user first selects it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:  # annotations only: the feature contract takes options, it does not own them
    from .options import FeatureOptions

__all__ = ["Feature"]

_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
_MIN_DEFINITION = 20


class Feature(ABC):
    """One image feature: identity, definition and one number from one image.

    Class attributes
    ----------------
    name:
        Stable column name, snake_case, and the module's own filename. It is
        the public identifier: ``extract_features(...)["hole_count"]``.
    definition:
        One human-auditable sentence, shown as a tooltip and used as the
        feature's documentation. Longer reasoning belongs in the module
        docstring.
    is_count:
        True for integer-valued counts (peaks, holes, crossings). Such values
        must be stable under small shifts and jitter; the stability tests
        assert exact equality for them.
    needs_colour:
        True if the feature reads absolute colour instead of the preprocessed
        luminance image. The pipeline is a property of the feature: it must
        not be inherited from whatever group the feature happens to sit in.
    """

    name: ClassVar[str]
    definition: ClassVar[str]
    is_count: ClassVar[bool] = False
    needs_colour: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not _NAME.match(name):
            raise TypeError(f"{cls.__name__}.name must be a snake_case identifier, got {name!r}")
        module = cls.__module__.rsplit(".", 1)[-1]
        if module != name:
            raise TypeError(
                f"{cls.__name__}: feature {name!r} must live in {name}.py, not {module}.py"
            )
        definition = getattr(cls, "definition", None)
        if (
            not isinstance(definition, str)
            or len(definition) < _MIN_DEFINITION
            or not definition[0].isupper()
            or not definition.endswith(".")
        ):
            raise TypeError(
                f"{cls.__name__}.definition must be a capitalised sentence ending in '.'"
            )
        for flag in ("is_count", "needs_colour"):
            if not isinstance(getattr(cls, flag), bool):
                raise TypeError(f"{cls.__name__}.{flag} must be a bool")

    @abstractmethod
    def compute(self, img: np.ndarray, opts: FeatureOptions) -> float:
        """The feature value for one preprocessed image.

        ``img`` is the luminance image for ``needs_colour = False`` features and
        the absolute colour image for ``needs_colour = True`` ones.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
