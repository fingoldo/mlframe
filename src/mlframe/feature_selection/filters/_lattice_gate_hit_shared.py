"""Shared ``@property`` bodies for the ``LatticeHit`` (``_integer_lattice_fe.py``) / ``ArgmaxHit``,
``GateHit`` (``_conditional_gate_fe.py``) frozen-dataclass hit records: independently duplicated
across those modules, consolidated here so a fix can't silently drift out of sync across copies.
Each module keeps its own module-local ``_responded`` (different ``_MIN_MARGIN``/``_MIN_NULL_MARGIN``
tuning is possible even though the current defaults match) - ``responded_property`` is a factory that
binds a class's ``responded`` property to whichever ``_responded`` the caller's module defines.
"""
from __future__ import annotations

from typing import Callable


def margin_over_operands(self) -> float:
    """MI gained by the engineered feature over the best raw operand / existing-op floor alone."""
    return float(self.feat_mi - self.operand_floor)


def responded_property(responded_fn: Callable[[float, float, float], bool]) -> property:
    """Bind a ``responded`` property to ``responded_fn(feat_mi, operand_floor, null_hi)`` - use as
    ``responded = responded_property(_responded)`` inside a frozen-dataclass body."""

    def _responded_prop(self) -> bool:
        """Evaluate the bound ``responded_fn`` on this hit's ``(feat_mi, operand_floor, null_hi)``."""
        return bool(responded_fn(self.feat_mi, self.operand_floor, self.null_hi))

    return property(_responded_prop)
