"""Builder and replay of the ``unary_credit`` recipe kind: one unary transform of one numeric column that the linear-usability credit (``_unary_credit_stats``) found useful, and of the ``unary_credit_product``
kind: the product of two such columns (each in its credited form, or the identity) that the interaction credit (``_unary_credit_interaction``) found, ``mul(sin(x1),log(x2))``.

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


def _form_extra(prefix: str, form: Any) -> "dict[str, Any]":
    """The ``extra`` entries of one side of a product recipe: the form name and its constants, or ``identity``."""
    if form is None:
        return {f"{prefix}form": "identity"}
    out: "dict[str, Any]" = {f"{prefix}form": str(form.name), f"{prefix}median": float(form.median), f"{prefix}scale": float(form.scale)}
    if form.grid_values:
        out[f"{prefix}grid_values"] = tuple(float(v) for v in form.grid_values)
        out[f"{prefix}grid_scores"] = tuple(float(v) for v in form.grid_scores)
    return out


def build_unary_credit_product_recipe(*, name: str, src_a: str, form_a: Any, src_b: str, form_b: Any) -> "EngineeredRecipe":
    """Frozen recipe for ``f_a(x_a) * f_b(x_b)``; a form that is ``None`` is the identity."""
    from . import EngineeredRecipe

    return EngineeredRecipe(name=name, kind="unary_credit_product", src_names=(str(src_a), str(src_b)), extra={**_form_extra("a_", form_a), **_form_extra("b_", form_b)})


def _side_values(recipe: "EngineeredRecipe", X: Any, index: int, prefix: str) -> np.ndarray:
    """One factor of a product recipe replayed on ``X``."""
    from . import _extract_column
    from .._unary_credit_stats import UnaryForm, apply_form

    extra = recipe.extra
    column = _extract_column(X, recipe.src_names[index])
    try:
        values = np.asarray(column, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unary_credit_product recipe '{recipe.name}': source column {recipe.src_names[index]!r} has non-numeric dtype {getattr(column, 'dtype', type(column).__name__)}") from exc
    if f"{prefix}form" not in extra:
        raise KeyError(f"unary_credit_product recipe '{recipe.name}' is missing {prefix + 'form'!r} in extra. Re-fit MRMR to regenerate.")
    if extra[f"{prefix}form"] == "identity":
        return values
    form = UnaryForm(
        str(extra[f"{prefix}form"]),
        float(extra[f"{prefix}median"]),
        float(extra[f"{prefix}scale"]),
        tuple(extra.get(f"{prefix}grid_values", ())),
        tuple(extra.get(f"{prefix}grid_scores", ())),
    )
    return apply_form(form, values)


def _apply_unary_credit_product(recipe: "EngineeredRecipe", X: Any) -> np.ndarray:
    """Replay a credited product on the two source columns of ``X``."""
    if len(recipe.src_names) != 2:
        raise ValueError(f"unary_credit_product recipe '{recipe.name}': src_names must have length 2; got {len(recipe.src_names)}")
    return _side_values(recipe, X, 0, "a_") * _side_values(recipe, X, 1, "b_")
