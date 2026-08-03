"""Shared numba-or-passthrough decorator for ``training/composite/discovery``'s causal-basis
kernels (``_grouped_causal_bases.py`` / ``_base_engineering.py``): independently duplicated
across those modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Callable, cast

try:  # numba is a core mlframe dep; the pure-numpy fallback keeps the module importable in a stripped CI env.
    import numba

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover -- numba always present in prod
    _HAVE_NUMBA = False


def njit_or_passthrough(func: Callable) -> Callable:
    """Apply ``numba.njit(cache=True)`` when available, else return the plain Python function (correctness-preserving)."""
    if _HAVE_NUMBA:
        return cast(Callable, numba.njit(cache=True)(func))
    return func
