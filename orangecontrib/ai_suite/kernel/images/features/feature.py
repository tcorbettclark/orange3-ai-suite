"""The feature contract: one class, one number, one description.

A feature is a class, not a module, so its required surface is explicit and
checked when the class is created. Subclasses declare ``name`` and
``description``, and optionally ``is_count`` / ``needs_colour``, then implement
:meth:`compute`.

The checks below run at import time, which is deliberate: a malformed feature
should fail when the package loads, not when a user first selects it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .description import Description

if TYPE_CHECKING:  # annotations only: the feature contract takes options, it does not own them
    from .options import FeatureOptions

__all__ = ["Feature"]

_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class Feature(ABC):
    """One image feature: identity, description and one number from one image.

    Class attributes
    ----------------
    name:
        Stable column name, snake_case, and the module's own filename. It is
        the public identifier: ``extract_features(...)["hole_count"]``.
    description:
        A :class:`~.description.Description`: the title a person reads, the
        one-sentence summary, the plain-language intuition and the technical
        account. Its own validation runs when the class is created.
    is_count:
        True when the value is a whole number: peaks and holes. Level counts
        that are averaged over lines are not whole numbers and do not carry it,
        which makes the flag something a caller can act on - a table column to
        treat as a count, a histogram to bin at integer edges.
    needs_colour:
        True if the feature reads absolute colour instead of the preprocessed
        luminance image. The pipeline is a property of the feature: it must
        not be inherited from whatever group the feature happens to sit in.
    """

    name: ClassVar[str]
    description: ClassVar[Description]
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
        if not isinstance(getattr(cls, "description", None), Description):
            raise TypeError(f"{cls.__name__}.description must be a Description")
        for flag in ("is_count", "needs_colour"):
            if not isinstance(getattr(cls, flag), bool):
                raise TypeError(f"{cls.__name__}.{flag} must be a bool")

    @abstractmethod
    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """The feature value for one preprocessed image.

        ``image`` is the luminance image for ``needs_colour = False`` features and
        the absolute colour image for ``needs_colour = True`` ones.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
