"""Builder and replay of the ``unary_credit`` recipe kind: one unary transform of one numeric column that the linear-usability credit (``_unary_credit_stats``) found useful.

The recipe carries the form name and the constants fitted on the discovery rows (median and scale of the column, the normal-score table of ``rankgauss``), so ``transform()`` rebuilds the column from the source column
alone: no reference to the target, leakage-free by construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from . import EngineeredRecipe


def build_unary_credit_recipe(*, name: str, src_name: str, form: Any) -> "EngineeredRecipe":
    """Frozen recipe for one credited unary form (``form`` is a ``_unary_credit_stats.UnaryForm``)."""
    from . import EngineeredRecipe

    extra = {"form": str(form.name), "median": float(form.median), "scale": float(form.scale)}
    if form.grid_values:
        extra["grid_values"] = tuple(float(v) for v in form.grid_values)
        extra["grid_scores"] = tuple(float(v) for v in form.grid_scores)
    return EngineeredRecipe(name=name, kind="unary_credit", src_names=(str(src_name),), extra=extra)


def _apply_unary_credit(recipe: "EngineeredRecipe", X: Any) -> np.ndarray:
    """Replay a credited unary form on the source column of ``X``."""
    from . import _extract_column
    from .._unary_credit_stats import UnaryForm, apply_form

    extra = recipe.extra
    for key in ("form", "median", "scale"):
        if key not in extra:
            raise KeyError(f"unary_credit recipe '{recipe.name}' is missing {key!r} in extra. Re-fit MRMR to regenerate.")
    if len(recipe.src_names) != 1:
        raise ValueError(f"unary_credit recipe '{recipe.name}': src_names must have length 1; got {len(recipe.src_names)}")
    column = _extract_column(X, recipe.src_names[0])
    try:
        values = np.asarray(column, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unary_credit recipe '{recipe.name}': source column {recipe.src_names[0]!r} has non-numeric dtype {getattr(column, 'dtype', type(column).__name__)}") from exc
    form = UnaryForm(
        str(extra["form"]), float(extra["median"]), float(extra["scale"]), tuple(extra.get("grid_values", ())), tuple(extra.get("grid_scores", ())),
    )
    return apply_form(form, values)
