"""Layer 77 (2026-06-01): recipe builder + apply for ``orth_quadruplet_cross``.

Sibling to ``engineered_recipes`` that hosts the closed-form replay logic
for quadruplet-cross-basis columns produced by
``_orthogonal_quadruplet_fe.hybrid_orth_mi_quadruplet_fe`` (and the
recipe-emitting wrapper).

Mirror of Layer 56 ``orth_triplet_cross`` with a fourth leg. Replay reads
only X -- no y at transform time, so ``MRMR.transform`` is leakage-free
by construction.

extra layout:
* orth_quadruplet_cross : {basis_i, basis_j, basis_k, basis_l,
                           deg_a, deg_b, deg_c, deg_d}
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe


def _apply_orth_quadruplet_cross(recipe: "EngineeredRecipe", X: Any) -> np.ndarray:
    """Replay a quadruplet-cross-basis column: extract four source columns,
    evaluate
    ``basis_a^{deg_a}(z_i) * basis_b^{deg_b}(z_j)
       * basis_c^{deg_c}(z_k) * basis_d^{deg_d}(z_l)``.
    Stateless given the stored bases + degrees; no y reference.
    """
    # Lazy import to avoid a circular dependency with ``engineered_recipes``.
    from .engineered_recipes import _eval_orth_basis_column, _extract_column

    if len(recipe.src_names) != 4:
        raise ValueError(
            f"orth_quadruplet_cross recipe '{recipe.name}' must have exactly 4 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in (
        "basis_i", "basis_j", "basis_k", "basis_l",
        "deg_a", "deg_b", "deg_c", "deg_d",
    ):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_quadruplet_cross recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    name_i, name_j, name_k, name_l = recipe.src_names
    basis_i = str(recipe.extra["basis_i"])
    basis_j = str(recipe.extra["basis_j"])
    basis_k = str(recipe.extra["basis_k"])
    basis_l = str(recipe.extra["basis_l"])
    deg_a = int(recipe.extra["deg_a"])
    deg_b = int(recipe.extra["deg_b"])
    deg_c = int(recipe.extra["deg_c"])
    deg_d = int(recipe.extra["deg_d"])
    # REPLAY-FIDELITY FIX (2026-06-13): per-leg FROZEN fit-time basis-preprocess params (mirrors the
    # pair "BUG2 FIX"); without them _eval_orth_basis_column refits z-score/min-max from APPLY-time rows,
    # silently shifting the basis axis on a row-slice / drifted test frame. None -> legacy refit path.
    pp_i = recipe.extra.get("preprocess_params_i")
    pp_j = recipe.extra.get("preprocess_params_j")
    pp_k = recipe.extra.get("preprocess_params_k")
    pp_l = recipe.extra.get("preprocess_params_l")
    vals_i = _extract_column(X, name_i)
    vals_j = _extract_column(X, name_j)
    vals_k = _extract_column(X, name_k)
    vals_l = _extract_column(X, name_l)
    h_a = _eval_orth_basis_column(vals_i, basis_i, deg_a, preprocess_params=pp_i)
    h_b = _eval_orth_basis_column(vals_j, basis_j, deg_b, preprocess_params=pp_j)
    h_c = _eval_orth_basis_column(vals_k, basis_k, deg_c, preprocess_params=pp_k)
    h_d = _eval_orth_basis_column(vals_l, basis_l, deg_d, preprocess_params=pp_l)
    return h_a * h_b * h_c * h_d


def build_orth_quadruplet_cross_recipe(
    *, name: str,
    src_a_name: str, src_b_name: str, src_c_name: str, src_d_name: str,
    basis_i: str, basis_j: str, basis_k: str, basis_l: str,
    deg_a: int, deg_b: int, deg_c: int, deg_d: int,
    preprocess_params_i: "dict | None" = None,
    preprocess_params_j: "dict | None" = None,
    preprocess_params_k: "dict | None" = None,
    preprocess_params_l: "dict | None" = None,
) -> "EngineeredRecipe":
    """Frozen recipe for one quadruplet cross-basis column
    ``basis_i^{deg_a}(preprocess(X[a])) * basis_j^{deg_b}(preprocess(X[b]))
       * basis_k^{deg_c}(preprocess(X[c])) * basis_l^{deg_d}(preprocess(X[d]))``.
    """
    from .engineered_recipes import EngineeredRecipe, _freeze_preprocess_params

    extra = {
        "basis_i": str(basis_i),
        "basis_j": str(basis_j),
        "basis_k": str(basis_k),
        "basis_l": str(basis_l),
        "deg_a": int(deg_a),
        "deg_b": int(deg_b),
        "deg_c": int(deg_c),
        "deg_d": int(deg_d),
    }
    # REPLAY-FIDELITY FIX (2026-06-13): freeze each leg's fit-time basis-preprocess params (no
    # slice-vs-full refit drift). Omitted when None so legacy recipes stay byte-equal.
    for _key, _pp in (("preprocess_params_i", preprocess_params_i),
                      ("preprocess_params_j", preprocess_params_j),
                      ("preprocess_params_k", preprocess_params_k),
                      ("preprocess_params_l", preprocess_params_l)):
        _frozen = _freeze_preprocess_params(_pp)
        if _frozen is not None:
            extra[_key] = _frozen
    return EngineeredRecipe(
        name=name,
        kind="orth_quadruplet_cross",
        src_names=(src_a_name, src_b_name, src_c_name, src_d_name),
        extra=extra,
    )
