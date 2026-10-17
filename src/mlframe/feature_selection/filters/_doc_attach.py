"""``set_doc``: attach a long docstring held in a constant to a function, keeping the function body within the function-length gate."""

from __future__ import annotations

from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., object])


def set_doc(doc: str) -> Callable[[_F], _F]:
    """Return a decorator that assigns ``doc`` as the wrapped function's ``__doc__`` and returns the function unchanged."""

    def _apply(fn: _F) -> _F:
        """Set ``fn.__doc__`` and return ``fn``."""
        fn.__doc__ = doc
        return fn

    return _apply
