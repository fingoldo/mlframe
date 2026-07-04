"""Replay + builders for missingness, pairwise-ratio, and MI-greedy recipe kinds.

The ``_apply_*`` for missingness / pairwise-ratio live in ``_missingness_fe`` /
``_ratio_delta_fe``; these are the thin ``build_*`` constructors plus the
self-contained ``_apply_mi_greedy_transform`` replay. The ``apply_recipe``
dispatcher lazy-imports the mi-greedy apply; builders + apply lazy-import the
parent's ``EngineeredRecipe`` / ``_extract_column`` in-body to avoid a cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from . import EngineeredRecipe


def _apply_mi_greedy_transform(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a generic MI-greedy engineered column. Stateless given the
    stored transform name + source columns; no y reference."""
    from . import _extract_column
    if "transform" not in recipe.extra:
        raise KeyError(
            f"mi_greedy_transform recipe '{recipe.name}' missing 'transform' "
            f"in extra. Re-fit MRMR to regenerate."
        )
    if len(recipe.src_names) not in (1, 2):
        raise ValueError(
            f"mi_greedy_transform recipe '{recipe.name}': src_names must have "
            f"length 1 (unary) or 2 (binary); got {len(recipe.src_names)}"
        )
    # Lazy import to avoid the circular dependency
    # (_mi_greedy_fe -> engineered_recipes via recipe builder).
    from .._mi_greedy_fe import apply_mi_greedy_transform

    def _numeric_source(n: str) -> np.ndarray:
        col = _extract_column(X, n)
        try:
            return np.asarray(col, dtype=np.float64)
        except (ValueError, TypeError) as e:
            # The mi_greedy registry is numeric-only; a non-numeric source otherwise raises deep in np.asarray at
            # TRANSFORM time only (fit/serve asymmetry). Surface a clear, recipe-scoped error naming the column.
            raise ValueError(
                f"mi_greedy_transform recipe '{recipe.name}': source column {n!r} has non-numeric dtype "
                f"{getattr(col, 'dtype', type(col).__name__)} that cannot be coerced to float."
            ) from e

    src_values = [_numeric_source(n) for n in recipe.src_names]
    return apply_mi_greedy_transform(str(recipe.extra["transform"]), src_values)


def build_mi_greedy_transform_recipe(
    *, name: str, transform: str, src_names: tuple[str, ...],
) -> EngineeredRecipe:
    """Frozen recipe for one generic MI-greedy engineered column. ``transform``
    must be one of the keys in ``_mi_greedy_fe.UNARY_TRANSFORMS`` (or
    ``TRIG_BOUNDED_TRANSFORMS``) for unary recipes, or
    ``_mi_greedy_fe.BINARY_TRANSFORMS`` for binary recipes; the registry
    lookup at replay time enforces the validation."""
    from . import EngineeredRecipe
    return EngineeredRecipe(
        name=name,
        kind="mi_greedy_transform",
        src_names=tuple(src_names),
        extra={"transform": str(transform)},
    )





# ---------------------------------------------------------------------------
# Layer 37 (2026-05-31): missingness-aware FE recipes (thin builders; the
# apply helpers live in ``_missingness_fe.py`` to keep this module under the
# 1k-LOC ceiling).
# ---------------------------------------------------------------------------


def build_missing_indicator_recipe(
    *, name: str, src_name: str,
) -> EngineeredRecipe:
    """Frozen recipe for one ``is_missing__{col}`` indicator. Stateless --
    replay just runs ``isna()`` on the source column."""
    from . import EngineeredRecipe
    return EngineeredRecipe(
        name=name,
        kind="missing_indicator",
        src_names=(src_name,),
        extra={},
    )


def build_missingness_count_recipe(
    *, name: str, cols: tuple,
) -> EngineeredRecipe:
    """Frozen recipe for the per-row missingness count across ``cols``.
    Replay re-counts ``isna()`` over the same columns; missing columns at
    test time contribute 0 (graceful schema-drift contract)."""
    from . import EngineeredRecipe
    cols_t = tuple(str(c) for c in cols)
    return EngineeredRecipe(
        name=name,
        kind="missingness_count",
        # src_names is the column subset (variadic): apply helpers read
        # from recipe.extra['cols'] so the dispatcher's invariants stay
        # uniform with the other kinds.
        src_names=cols_t,
        extra={"cols": cols_t},
    )


def build_missingness_pattern_recipe(
    *, name: str, cols: tuple, pattern_to_label: dict,
    other_label: int, top_k: int,
) -> EngineeredRecipe:
    """Frozen recipe for the per-row top-K pattern label. The pattern
    signature dict maps the bit-packed isna signature to an integer
    label; unseen signatures at transform map to ``other_label``."""
    from . import EngineeredRecipe
    cols_t = tuple(str(c) for c in cols)
    # Coerce keys to int for stable pickle round-trip (signatures are
    # int64 bit-packs of the isna mask).
    pattern_clean = {int(k): int(v) for k, v in pattern_to_label.items()}
    return EngineeredRecipe(
        name=name,
        kind="missingness_pattern",
        src_names=cols_t,
        extra={
            "cols": cols_t,
            "pattern_to_label": pattern_clean,
            "other_label": int(other_label),
            "top_k": int(top_k),
        },
    )


# ---------------------------------------------------------------------------
# Layer 38 (2026-05-31): cross-feature ratio + grouped-delta + lagged-diff
# (thin builders; apply helpers live in ``_ratio_delta_fe.py``).
# ---------------------------------------------------------------------------


def build_pairwise_ratio_recipe(
    *, name: str, src_a_name: str, src_b_name: str,
    kind: str = "div", eps: float = 1e-9,
) -> EngineeredRecipe:
    """Frozen recipe for ``a / b`` (``kind='div'``, safe-division floored at
    ``eps``) or ``log1p(|a|+eps) - log1p(|b|+eps)`` (``kind='log_div'``)."""
    from . import EngineeredRecipe
    if kind not in ("div", "log_div"):
        raise ValueError(
            f"pairwise_ratio kind must be 'div' or 'log_div'; got {kind!r}"
        )
    return EngineeredRecipe(
        name=name,
        kind="pairwise_ratio",
        src_names=(src_a_name, src_b_name),
        extra={"kind": str(kind), "eps": float(eps)},
    )


