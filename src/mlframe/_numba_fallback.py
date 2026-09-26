"""The one no-op stand-in for ``numba.njit``, used by modules whose ``from numba import njit`` sits in a try/except.

numba is a hard dependency, so this runs only in a stripped environment; a single copy keeps the fallback modules from
drifting apart (sixteen hand-written copies had, in which call forms they accepted).
"""

from __future__ import annotations

from typing import Any, Callable


def njit(*args: Any, **kwargs: Any) -> Any:
    """Return the function unchanged for ``@njit`` and ``@njit(...)`` alike, so decorated kernels run as plain Python."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(fn: Callable) -> Callable:
        """Identity decorator for the ``@njit(...)`` call form."""
        return fn

    return deco
