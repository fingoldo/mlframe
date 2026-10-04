"""Budget parameters (wall-clock minutes/seconds, refit or iteration caps): one reading for the whole package.

Convention: ``None`` and ``0`` both mean NO LIMIT; a non-positive value is never a budget of zero. Every budget check goes through
:func:`active_budget`, so the convention is written once instead of as a bare truth test at each site, where ``0`` reads as "absent"
only by accident and a later ``is not None`` rewrite would silently turn it into "stop immediately".
"""

from __future__ import annotations

from typing import Optional, TypeVar

__all__ = ["active_budget"]

_N = TypeVar("_N", int, float)


def active_budget(value: Optional[_N]) -> Optional[_N]:
    """``value`` when it is a positive limit, else ``None`` (no limit): ``None``, ``0`` and negatives all mean unlimited.

    >>> active_budget(None) is None, active_budget(0) is None, active_budget(0.0) is None, active_budget(-1) is None
    (True, True, True, True)
    >>> active_budget(5), active_budget(0.5)
    (5, 0.5)
    """
    if value is None or value <= 0:
        return None
    return value
