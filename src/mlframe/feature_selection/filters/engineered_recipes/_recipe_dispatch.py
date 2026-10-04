"""The ``apply_recipe`` dispatcher: route an ``EngineeredRecipe`` to its replay helper.

The per-kind ``_apply_*`` helpers live in cohesive sibling submodules (numeric
pair, factorize, hermite/cluster/target-encoding) and in the heavier FE siblings
one package level up. Each kind has a small handler in ``_KIND_HANDLERS`` that
lazy-imports its helper at call time, so this dispatcher stays a thin,
dependency-light routing table with no top-level import cycle against the
appliers (which themselves recurse back here for nested-engineered operands).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, cast

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _extract_column

_Handler = Callable[[EngineeredRecipe, Any, "dict[str, np.ndarray] | None", "dict[tuple, np.ndarray] | None"], np.ndarray]


def _routed(module: str, func: str, *, col_cache: bool = False, basis_cache: bool = False) -> _Handler:
    """Handler that lazy-imports ``func`` from ``module`` (relative to this package) and calls it as ``func(recipe, X[, col_cache=][, basis_cache=])``.

    The import happens at call time, inside the handler, exactly where the original if-chain imported it, so a patch of the helper on its own module
    is seen and the module keeps no top-level import of the heavier FE siblings (some of which import this dispatcher back).
    """

    def _handler(recipe: EngineeredRecipe, X: Any, cc: "dict[str, np.ndarray] | None", bc: "dict[tuple, np.ndarray] | None") -> np.ndarray:
        """Resolve the helper now and replay ``recipe`` with the caches this kind accepts."""
        helper = getattr(importlib.import_module(module, __package__), func)
        kwargs: dict[str, Any] = {}
        if col_cache:
            kwargs["col_cache"] = cc
        if basis_cache:
            kwargs["basis_cache"] = bc
        return cast(np.ndarray, helper(recipe, X, **kwargs))

    return _handler


def _missingness(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 37 lazy import: the helpers live alongside the encoders that build them."""
    from .._missingness_fe import (
        _apply_missing_indicator_recipe,
        _apply_missingness_count_recipe,
        _apply_missingness_pattern_recipe,
    )

    if recipe.kind == "missing_indicator":
        return _apply_missing_indicator_recipe(recipe, X)
    if recipe.kind == "missingness_count":
        return _apply_missingness_count_recipe(recipe, X)
    return _apply_missingness_pattern_recipe(recipe, X)


def _ratio_delta(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 38 lazy import: the replay helpers live with the ratio / delta generators."""
    from .._ratio_delta_fe import (
        _apply_pairwise_ratio_recipe,
        _apply_grouped_delta_recipe,
        _apply_lagged_diff_recipe,
    )

    if recipe.kind == "pairwise_ratio":
        return _apply_pairwise_ratio_recipe(recipe, X)
    if recipe.kind == "grouped_delta":
        return _apply_grouped_delta_recipe(recipe, X)
    return _apply_lagged_diff_recipe(recipe, X)


def _cat_cross(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 89 / 94 lazy import: cat x cat (and cat x cat x cat) synergy cross; the replay helper lives with the generator.

    A pandas frame is passed through untouched; any other container is rebuilt as a frame of just the source columns.
    """
    names = tuple(recipe.src_names)
    frame = X if (pd is not None and isinstance(X, pd.DataFrame)) else pd.DataFrame({n: _extract_column(X, n, col_cache=cc) for n in names})
    mapping = {tuple(k): int(v) for k, v in recipe.extra["mapping"]}
    extras: dict[str, Any] = {
        "encoding": str(recipe.extra.get("encoding", "raw")),
        "te_lookup": {int(k): float(v) for k, v in recipe.extra.get("te_lookup", [])},
        "global_mean": float(recipe.extra.get("global_mean", 0.0)),
    }
    if recipe.kind == "cat_pair_cross":
        from .._cat_pair_fe import apply_cat_pair_cross

        cat_i, cat_j = names
        return apply_cat_pair_cross(frame, cat_i, cat_j, mapping, **extras)
    from .._cat_triple_fe import apply_cat_triple_cross

    cat_a, cat_b, cat_c = names
    return apply_cat_triple_cross(frame, cat_a, cat_b, cat_c, mapping, **extras)


def _numeric_decompose(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 90: numeric decomposition (multi-precision rounding + decimal-digit extraction).

    Pure arithmetic on the single source column - no y reference.
    """
    vals = _extract_column(X, recipe.src_names[0], col_cache=cc)
    if recipe.kind == "numeric_rounding":
        from .._numeric_decompose_fe import apply_rounding

        return apply_rounding(vals, float(recipe.extra["precision"]))
    from .._numeric_decompose_fe import apply_digit_extract

    return apply_digit_extract(vals, int(recipe.extra["digit_position"]))


def _modular(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 95 PART A: periodic / modular decomposition (x mod period + sin/cos phase) on the single source column; no y reference."""
    from .._periodic_fe import apply_modular

    vals = _extract_column(X, recipe.src_names[0], col_cache=cc)
    return apply_modular(vals, float(recipe.extra["period"]), str(recipe.extra["op"]))


def _pairwise_modular(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Pairwise / n-way modular residue: combine the source columns (sum/diff/prod/sum3/self) then take mod modulus.

    Pure integer arithmetic on X, no y reference -> leak-free, train/test exact.
    """
    from .._pairwise_modular_fe import apply_pairwise_modular

    return apply_pairwise_modular(X, str(recipe.extra["op"]), recipe.src_names, int(recipe.extra["modulus"]))


def _integer_lattice(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Pairwise integer-lattice column: cast both source columns to int then apply gcd / lcm / bitwise_and.

    Pure integer arithmetic on X, no y reference -> leak-free, train/test exact.
    """
    from .._integer_lattice_fe import apply_integer_lattice

    return apply_integer_lattice(X, str(recipe.extra["op"]), recipe.src_names)


def _row_argmax(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Row-argmax: the integer index of the row-maximum over the source columns. Pure function of X, no y -> leak-free."""
    from .._conditional_gate_fe import apply_row_argmax

    return apply_row_argmax(X, recipe.src_names)


def _conditional_gate(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Conditional-gate: c>tau ? a : b (select) / 1[c>tau]*a (mask), with the FROZEN tau. Pure function of X, no y -> leak-free."""
    from .._conditional_gate_fe import apply_conditional_gate

    return apply_conditional_gate(X, str(recipe.extra["mode"]), recipe.src_names, float(recipe.extra["tau"]))


def _temporal(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 92: leak-safe temporal aggregations.

    Replay computes each test row's expanding / rolling / lag stat against the stored TRAIN per-entity history plus earlier within-test rows - never
    the row's own future, never train labels.
    """
    from .._temporal_agg_fe import (
        apply_temporal_expanding,
        apply_temporal_rolling,
        apply_temporal_lag,
    )

    if recipe.kind == "temporal_expanding":
        return apply_temporal_expanding(X, dict(recipe.extra))
    if recipe.kind == "temporal_rolling":
        return apply_temporal_rolling(X, dict(recipe.extra))
    return apply_temporal_lag(X, dict(recipe.extra))


def _grouped_quantile(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Layer 88 lazy import: per-group distributional FE (percentile-rank + spread + target-aware supervised bins); replay helpers live with the generator."""
    from .._grouped_quantile_fe import (
        _apply_grouped_quantile_recipe,
        _apply_target_aware_group_bin_recipe,
    )

    if recipe.kind == "grouped_quantile":
        return _apply_grouped_quantile_recipe(recipe, X)
    return _apply_target_aware_group_bin_recipe(recipe, X)


def _binned_numeric_agg(recipe: EngineeredRecipe, X: Any, cc: Any, bc: Any) -> np.ndarray:
    """Grouped aggregation over quantile-binned numeric cells: replay bins the raw group column through the stored quantile edges and gathers the per-cell
    statistic; reads only X."""
    from .._binned_numeric_agg_fe import apply_binned_numeric_agg

    return apply_binned_numeric_agg(X, dict(recipe.extra))


# kind -> handler. Notes kept from the branches this table replaced:
#   orth_diff_basis        Layer 59: the apply helper lives in the sibling FE module, which keeps this module under the LOC ceiling.
#   orth_cluster_basis     Layer 61: per-cluster shared-basis FE; replay recomputes the aggregate from the stored member tuple via the recipe-stored
#                          aggregator (mean_z / median_z / pc1), then evaluates the same basis_degree - bit-exact round-trip from fit to transform.
#   orth_quadruplet_cross  Layer 77: 4-way cross-basis FE; closed-form over the four source columns via the recipe-stored (basis, deg) tuple.
#   hinge_basis            Backlog #11: closed-form max(x-tau,0) / max(tau-x,0) / 1[x>tau] from the stored {tau, side}; a pure function of the source column.
#   orth_wavelet           Backlog #13: Haar wavelet; closed-form dyadic indicator psi_{j,k}(clip((x-lo)/span,0,1)) from the stored {j, k, lo, span}.
#   composite_group_agg    Layer 93: composite (multi-column) group-key aggregate; the replay helper lives with the generator.
#   group_distance         Layer 95 PART B: per-group distribution-distance FE; replay maps a row's group key through the stored per-group scalar lookup.
#   rare_category          Layer 104: rare-category indicator / frequency-band via the stored per-category frequency lookup.
#   conditional_residual   Layer 104: x_i - E[x_i | bin(x_j)] with the stored quantile edges and per-bin mean of x_i.
#   conditional_dispersion Family D: conditional z-score / |z| / z^2 from the stored per-bin (mu_hat, sigma_hat) of x_i.
#   rankgauss              Layer 104: interpolates each test value's rank against the stored sorted fit values and maps to a Gaussian quantile.
_KIND_HANDLERS: dict[str, _Handler] = {
    "unary_binary": _routed("._recipe_unary_binary", "_apply_unary_binary", col_cache=True),
    "factorize": _routed("._recipe_factorize", "_apply_factorize", col_cache=True),
    "target_encoding": _routed("._recipe_poly_cluster", "_apply_target_encoding", col_cache=True),
    "hermite_pair": _routed("._recipe_poly_cluster", "_apply_hermite_pair", col_cache=True),
    "cluster_aggregate": _routed("._recipe_poly_cluster", "_apply_cluster_aggregate", col_cache=True),
    "orth_univariate": _routed("._orth_basis_recipes", "_apply_orth_univariate", col_cache=True, basis_cache=True),
    "orth_pair_cross": _routed("._orth_basis_recipes", "_apply_orth_pair_cross", col_cache=True, basis_cache=True),
    "orth_diff_basis": _routed(".._orthogonal_diff_basis_fe", "_apply_orth_diff_basis"),
    "orth_cluster_basis": _routed(".._orthogonal_cluster_basis_fe", "_apply_orth_cluster_basis"),
    "orth_triplet_cross": _routed(".._orthogonal_triplet_fe_recipes", "_apply_orth_triplet_cross"),
    "orth_quadruplet_cross": _routed(".._orthogonal_quadruplet_fe_recipes", "_apply_orth_quadruplet_cross"),
    "orth_spline": _routed("._orth_basis_recipes", "_apply_orth_spline", col_cache=True),
    "orth_fourier": _routed("._orth_basis_recipes", "_apply_orth_fourier", col_cache=True),
    "hinge_basis": _routed(".._hinge_basis_fe", "_apply_hinge_basis"),
    "orth_wavelet": _routed(".._wavelet_basis_fe_recipes", "_apply_orth_wavelet"),
    "mi_greedy_transform": _routed("._missingness_ratio_recipes", "_apply_mi_greedy_transform"),
    "kfold_target_encoded": _routed("._encoding_recipes", "_apply_kfold_target_encoded", col_cache=True),
    "count_encoded": _routed("._encoding_recipes", "_apply_count_encoded", col_cache=True),
    "frequency_encoded": _routed("._encoding_recipes", "_apply_frequency_encoded", col_cache=True),
    "cat_num_residual": _routed("._encoding_recipes", "_apply_cat_num_residual", col_cache=True),
    "missing_indicator": _missingness,
    "missingness_count": _missingness,
    "missingness_pattern": _missingness,
    "pairwise_ratio": _ratio_delta,
    "grouped_delta": _ratio_delta,
    "lagged_diff": _ratio_delta,
    "grouped_agg": _routed(".._grouped_agg_fe", "_apply_grouped_agg_recipe"),
    "composite_group_agg": _routed(".._composite_group_agg_fe", "_apply_composite_group_agg_recipe"),
    "cat_pair_cross": _cat_cross,
    "cat_triple_cross": _cat_cross,
    "numeric_rounding": _numeric_decompose,
    "digit_extract": _numeric_decompose,
    "modular": _modular,
    "pairwise_modular": _pairwise_modular,
    "pairwise_integer_lattice": _integer_lattice,
    "row_argmax": _row_argmax,
    "conditional_gate": _conditional_gate,
    "group_distance": _routed(".._group_distance_fe", "_apply_group_distance_recipe"),
    "rare_category": _routed(".._extra_fe_families", "_apply_rare_category_recipe"),
    "conditional_residual": _routed(".._extra_fe_families", "_apply_conditional_residual_recipe"),
    "conditional_dispersion": _routed(".._extra_fe_families", "_apply_conditional_dispersion_recipe"),
    "rankgauss": _routed(".._extra_fe_families", "_apply_rankgauss_recipe"),
    "temporal_expanding": _temporal,
    "temporal_rolling": _temporal,
    "temporal_lag": _temporal,
    "grouped_quantile": _grouped_quantile,
    "target_aware_group_bin": _grouped_quantile,
    "binned_numeric_agg": _binned_numeric_agg,
    "conditional_quantile_rank": _routed(".._conditional_quantile_rank_fe", "_apply_conditional_quantile_rank_recipe"),
    "ordinal_pattern_te": _routed(".._ordinal_pattern_fe", "_apply_ordinal_pattern_te_recipe"),
    "random_fourier": _routed(".._random_fourier_features_fe", "_apply_random_fourier_recipe"),
    "sir_direction": _routed(".._sliced_inverse_regression_fe", "_apply_sir_direction_recipe"),
    "lof_score": _routed(".._lof_fe", "_apply_lof_recipe"),
    "mahalanobis_density": _routed(".._mahalanobis_density_fe", "_apply_mahalanobis_density_recipe"),
}


def apply_recipe(
    recipe: EngineeredRecipe, X: Any,
    col_cache: "dict[str, np.ndarray] | None" = None,
    basis_cache: "dict[tuple, np.ndarray] | None" = None,
) -> np.ndarray:
    """Replay ``recipe`` against ``X`` and return the engineered column as a 1-D ndarray. Output dtype matches the recipe's quantization dtype if
    discretized, else float32 (matching fit-time ``check_prospective_fe_pairs`` working buffer dtype). Hot path in ``transform()`` - keep allocation-light.

    ``col_cache`` / ``basis_cache``: OPTIONAL, caller-owned dicts scoped to ONE ``transform()``/``predict()`` call, shared across every recipe in that
    call's replay list. ``col_cache`` dedupes ``_extract_column`` (a hub source column referenced by many recipes is pulled from ``X`` once, not once
    per recipe); ``basis_cache`` dedupes the orth-basis polynomial evaluation for a hub operand shared by sibling ``orth_pair_cross``/``orth_univariate``
    recipes. Both default to ``None`` (current always-recompute behaviour) so every EXISTING caller (none of which construct these dicts yet) is
    unaffected; a caller opts in by passing the SAME dict across its whole recipe-list replay loop."""
    handler = _KIND_HANDLERS.get(recipe.kind)
    if handler is None:
        raise ValueError(f"Unknown recipe kind: {recipe.kind!r}")
    return handler(recipe, X, col_cache, basis_cache)
