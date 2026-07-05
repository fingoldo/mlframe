"""Layer 56 (2026-05-31): recipe builder + apply for ``orth_triplet_cross``.

Sibling to ``engineered_recipes`` that hosts the closed-form replay logic
for triplet-cross-basis columns produced by
``_orthogonal_triplet_fe.hybrid_orth_mi_triplet_fe`` (and the recipe-
emitting wrapper).

Mirror of the Layer 22 ``orth_pair_cross`` recipe (``engineered_recipes._apply_orth_pair_cross``)
with a third leg. Replay reads only X -- no y at transform time, so
``MRMR.transform`` is leakage-free by construction.

extra layout:
* orth_triplet_cross : {basis_i, basis_j, basis_k, deg_a, deg_b, deg_c}
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe


def _apply_orth_triplet_cross(recipe: "EngineeredRecipe", X: Any) -> np.ndarray:
    """Replay a triplet-cross-basis column: extract three source columns,
    evaluate basis_a^{deg_a}(z_i) * basis_b^{deg_b}(z_j) * basis_c^{deg_c}(z_k).
    Stateless given the stored bases + degrees; no y reference.
    """
    # Lazy import to avoid a circular dependency: ``engineered_recipes`` imports
    # this module at its tail, and this module imports the basis evaluator + extractor
    # from there. Importing at call time decouples module-load order.
    from .engineered_recipes import _eval_orth_basis_column, _extract_column

    if len(recipe.src_names) != 3:
        raise ValueError(f"orth_triplet_cross recipe '{recipe.name}' must have exactly 3 " f"src_names; got {len(recipe.src_names)}")
    for key in ("basis_i", "basis_j", "basis_k", "deg_a", "deg_b", "deg_c"):
        if key not in recipe.extra:
            raise KeyError(f"orth_triplet_cross recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    name_i, name_j, name_k = recipe.src_names
    basis_i = str(recipe.extra["basis_i"])
    basis_j = str(recipe.extra["basis_j"])
    basis_k = str(recipe.extra["basis_k"])
    deg_a = int(recipe.extra["deg_a"])
    deg_b = int(recipe.extra["deg_b"])
    deg_c = int(recipe.extra["deg_c"])
    # REPLAY-FIDELITY FIX (2026-06-13): per-leg FROZEN fit-time basis-preprocess params, mirroring the
    # pair "BUG2 FIX". Without them ``_eval_orth_basis_column`` REFITS the z-score mean/std (or min-max
    # lo/hi) from the APPLY-time rows, so a row-slice / drifted test frame shifts the basis axis and
    # silently emits wrong values. ``None`` (legacy pickles) falls back to the refit path, byte-identical.
    pp_i = recipe.extra.get("preprocess_params_i")
    pp_j = recipe.extra.get("preprocess_params_j")
    pp_k = recipe.extra.get("preprocess_params_k")
    vals_i = _extract_column(X, name_i)
    vals_j = _extract_column(X, name_j)
    vals_k = _extract_column(X, name_k)
    h_a = _eval_orth_basis_column(vals_i, basis_i, deg_a, preprocess_params=pp_i)
    h_b = _eval_orth_basis_column(vals_j, basis_j, deg_b, preprocess_params=pp_j)
    h_c = _eval_orth_basis_column(vals_k, basis_k, deg_c, preprocess_params=pp_k)
    return h_a * h_b * h_c


def build_orth_triplet_cross_recipe(
    *, name: str,
    src_a_name: str, src_b_name: str, src_c_name: str,
    basis_i: str, basis_j: str, basis_k: str,
    deg_a: int, deg_b: int, deg_c: int,
    preprocess_params_i: "dict | None" = None,
    preprocess_params_j: "dict | None" = None,
    preprocess_params_k: "dict | None" = None,
) -> "EngineeredRecipe":
    """Frozen recipe for one triplet cross-basis column
    ``basis_i^{deg_a}(preprocess(X[a])) *
      basis_j^{deg_b}(preprocess(X[b])) *
      basis_k^{deg_c}(preprocess(X[c]))``.
    """
    from .engineered_recipes import EngineeredRecipe, _freeze_preprocess_params

    extra = {
        "basis_i": str(basis_i),
        "basis_j": str(basis_j),
        "basis_k": str(basis_k),
        "deg_a": int(deg_a),
        "deg_b": int(deg_b),
        "deg_c": int(deg_c),
    }
    # REPLAY-FIDELITY FIX (2026-06-13): freeze each leg's fit-time basis-preprocess params so replay
    # reproduces the axis byte-exactly (no slice-vs-full mean/std refit drift). Omitted when None so
    # legacy recipes stay byte-equal.
    for _key, _pp in (("preprocess_params_i", preprocess_params_i), ("preprocess_params_j", preprocess_params_j), ("preprocess_params_k", preprocess_params_k)):
        _frozen = _freeze_preprocess_params(_pp)
        if _frozen is not None:
            extra[_key] = _frozen
    return EngineeredRecipe(
        name=name,
        kind="orth_triplet_cross",
        src_names=(src_a_name, src_b_name, src_c_name),
        extra=extra,
    )
