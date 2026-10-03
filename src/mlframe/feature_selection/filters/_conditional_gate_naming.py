"""Engineered-column naming and frozen-recipe builders for the row-argmax and conditional-gate families.

Carved out of ``_conditional_gate_fe`` for file size: these are pure functions of their arguments with no dependency on the scan code.
Re-exported from the parent, so import sites are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

ROW_ARGMAX_PREFIX = "argmax"
CONDITIONAL_GATE_PREFIX = "gate"

GATE_MODES = ("select", "mask")


def engineered_name_row_argmax(cols: Sequence[str]) -> str:
    """Canonical engineered column name for one row-argmax, e.g. ``argmax__a__b__c``. Source columns join with ``__``."""
    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{ROW_ARGMAX_PREFIX}__{joined}"


def engineered_name_conditional_gate(mode: str, cols: Sequence[str], tau: float) -> str:
    """Canonical engineered column name for one conditional-gate column, e.g. ``gate_select__a__b__c__t0.123`` /
    ``gate_mask__a__c__t-0.4``. mode + source columns + the frozen tau fully determine the column."""
    if mode not in GATE_MODES:
        raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")
    joined = "__".join(str(c) for c in cols)
    return f"{CONDITIONAL_GATE_PREFIX}_{mode}__{joined}__t{float(tau):.6g}"


def build_row_argmax_recipe(*, name: str, cols: Sequence[str]) -> EngineeredRecipe:
    """Frozen recipe for one row-argmax column. Replay is ``np.argmax`` over the stacked source columns - no parameters."""
    from .engineered_recipes import EngineeredRecipe

    if len(cols) < 2:
        raise ValueError(f"row-argmax needs >= 2 source columns; got {tuple(cols)!r}")
    return EngineeredRecipe(name=name, kind="row_argmax", src_names=tuple(str(c) for c in cols))


def build_conditional_gate_recipe(*, name: str, mode: str, cols: Sequence[str], tau: float) -> EngineeredRecipe:
    """Frozen recipe for one conditional-gate column. The chosen ``tau`` is FROZEN in ``extra`` for exact replay."""
    from .engineered_recipes import EngineeredRecipe

    if mode not in GATE_MODES:
        raise ValueError(f"conditional-gate mode must be one of {GATE_MODES}; got {mode!r}")
    return EngineeredRecipe(
        name=name,
        kind="conditional_gate",
        src_names=tuple(str(c) for c in cols),
        extra={"mode": str(mode), "tau": float(tau)},
    )
