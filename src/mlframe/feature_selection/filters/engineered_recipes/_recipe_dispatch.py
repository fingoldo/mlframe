"""The ``apply_recipe`` dispatcher: route an ``EngineeredRecipe`` to its replay helper.

The per-kind ``_apply_*`` helpers live in cohesive sibling submodules (numeric
pair, factorize, hermite/cluster/target-encoding) and in the heavier FE siblings
one package level up. ``apply_recipe`` lazy-imports each helper inside its branch
so this dispatcher stays a thin, dependency-light routing table with no top-level
import cycle against the appliers (which themselves recurse back here for
nested-engineered operands).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _extract_column


def apply_recipe(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay ``recipe`` against ``X`` and return the engineered column as a 1-D ndarray. Output dtype matches the recipe's quantization dtype if
    discretized, else float32 (matching fit-time ``check_prospective_fe_pairs`` working buffer dtype). Hot path in ``transform()`` -- keep allocation-light."""
    if recipe.kind == "unary_binary":
        from ._recipe_unary_binary import _apply_unary_binary
        return _apply_unary_binary(recipe, X)
    if recipe.kind == "factorize":
        from ._recipe_factorize import _apply_factorize
        return _apply_factorize(recipe, X)
    if recipe.kind == "target_encoding":
        from ._recipe_poly_cluster import _apply_target_encoding
        return _apply_target_encoding(recipe, X)
    if recipe.kind == "hermite_pair":
        from ._recipe_poly_cluster import _apply_hermite_pair
        return _apply_hermite_pair(recipe, X)
    if recipe.kind == "cluster_aggregate":
        from ._recipe_poly_cluster import _apply_cluster_aggregate
        return _apply_cluster_aggregate(recipe, X)
    if recipe.kind == "orth_univariate":
        from ._orth_basis_recipes import _apply_orth_univariate
        return _apply_orth_univariate(recipe, X)
    if recipe.kind == "orth_pair_cross":
        from ._orth_basis_recipes import _apply_orth_pair_cross
        return _apply_orth_pair_cross(recipe, X)
    if recipe.kind == "orth_diff_basis":
        # Layer 59 (2026-05-31): lazy import keeps this module under the
        # ~1.8k-LOC ceiling; the apply helper lives in the sibling FE module.
        from .._orthogonal_diff_basis_fe import _apply_orth_diff_basis
        return _apply_orth_diff_basis(recipe, X)
    if recipe.kind == "orth_cluster_basis":
        # Layer 61 (2026-05-31): per-cluster shared-basis FE. Replay
        # recomputes the aggregate from the stored member tuple via the
        # recipe-stored aggregator (mean_z / median_z / pc1), then evaluates
        # the same basis_degree -- bit-exact round-trip from fit to transform.
        from .._orthogonal_cluster_basis_fe import _apply_orth_cluster_basis
        return _apply_orth_cluster_basis(recipe, X)
    if recipe.kind == "orth_triplet_cross":
        from .._orthogonal_triplet_fe_recipes import _apply_orth_triplet_cross
        return _apply_orth_triplet_cross(recipe, X)
    if recipe.kind == "orth_quadruplet_cross":
        # Layer 77 (2026-06-01): 4-way cross-basis FE. Replay is closed-form
        # over the four source columns via the recipe-stored (basis, deg) tuple.
        from .._orthogonal_quadruplet_fe_recipes import _apply_orth_quadruplet_cross
        return _apply_orth_quadruplet_cross(recipe, X)
    if recipe.kind == "orth_spline":
        from ._orth_basis_recipes import _apply_orth_spline
        return _apply_orth_spline(recipe, X)
    if recipe.kind == "orth_fourier":
        from ._orth_basis_recipes import _apply_orth_fourier
        return _apply_orth_fourier(recipe, X)
    if recipe.kind == "hinge_basis":
        # Backlog #11 (2026-06-09): hinge / piecewise-linear change-point basis.
        # Replay is closed-form ``max(x-tau,0)`` / ``max(tau-x,0)`` / ``1[x>tau]``
        # from the stored ``{tau, side}`` -- a pure function of the source column,
        # no y reference. Lazy import keeps this dispatcher dependency-light; the
        # apply helper lives with the hinge generator one package level up.
        from .._hinge_basis_fe import _apply_hinge_basis
        return _apply_hinge_basis(recipe, X)
    if recipe.kind == "orth_wavelet":
        # Backlog #13 (2026-06-09): Haar wavelet / localized multiresolution basis.
        # Replay is the closed-form dyadic indicator ``psi_{j,k}(clip((x-lo)/span,
        # 0,1))`` from the stored ``{j, k, lo, span}`` -- a pure function of the
        # source column, no y reference (structurally like orth_spline). Lazy
        # import keeps this dispatcher dependency-light; the apply helper lives
        # with the wavelet generator one package level up.
        from .._wavelet_basis_fe import _apply_orth_wavelet
        return _apply_orth_wavelet(recipe, X)
    if recipe.kind == "mi_greedy_transform":
        from ._missingness_ratio_recipes import _apply_mi_greedy_transform
        return _apply_mi_greedy_transform(recipe, X)
    if recipe.kind == "kfold_target_encoded":
        from ._encoding_recipes import _apply_kfold_target_encoded
        return _apply_kfold_target_encoded(recipe, X)
    if recipe.kind == "count_encoded":
        from ._encoding_recipes import _apply_count_encoded
        return _apply_count_encoded(recipe, X)
    if recipe.kind == "frequency_encoded":
        from ._encoding_recipes import _apply_frequency_encoded
        return _apply_frequency_encoded(recipe, X)
    if recipe.kind == "cat_num_residual":
        from ._encoding_recipes import _apply_cat_num_residual
        return _apply_cat_num_residual(recipe, X)
    if recipe.kind in ("missing_indicator", "missingness_count", "missingness_pattern"):
        # Layer 37 lazy import: keeps engineered_recipes.py under the 1k-LOC
        # ceiling (mlframe sibling-split rule). The helpers live alongside
        # the encoders that build them.
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
    if recipe.kind in ("pairwise_ratio", "grouped_delta", "lagged_diff"):
        # Layer 38 lazy import: same rationale as Layer 37 -- keep this module
        # under the 1k-LOC ceiling.
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
    if recipe.kind == "grouped_agg":
        # Layer 87 lazy import: keep this module under the LOC ceiling; the
        # apply helper lives alongside the grouped-agg generator.
        from .._grouped_agg_fe import _apply_grouped_agg_recipe
        return _apply_grouped_agg_recipe(recipe, X)
    if recipe.kind == "composite_group_agg":
        # Layer 93 lazy import: composite (multi-column) group-key aggregate;
        # replay helper lives with the generator. Keeps this module under the
        # LOC ceiling.
        from .._composite_group_agg_fe import _apply_composite_group_agg_recipe
        return _apply_composite_group_agg_recipe(recipe, X)
    if recipe.kind == "cat_pair_cross":
        # Layer 89 lazy import: cat x cat synergy cross; replay helper lives
        # with the generator. Keeps this module under the LOC ceiling.
        from .._cat_pair_fe import apply_cat_pair_cross
        cat_i, cat_j = recipe.src_names
        return apply_cat_pair_cross(
            X if (pd is not None and isinstance(X, pd.DataFrame))
            else pd.DataFrame({
                cat_i: _extract_column(X, cat_i),
                cat_j: _extract_column(X, cat_j),
            }),
            cat_i, cat_j,
            {tuple(k): int(v) for k, v in recipe.extra["mapping"]},
            encoding=str(recipe.extra.get("encoding", "raw")),
            te_lookup={
                int(k): float(v) for k, v in recipe.extra.get("te_lookup", [])
            },
            global_mean=float(recipe.extra.get("global_mean", 0.0)),
        )
    if recipe.kind == "cat_triple_cross":
        # Layer 94 lazy import: cat x cat x cat synergy cross; replay helper
        # lives with the generator. Keeps this module under the LOC ceiling.
        from .._cat_triple_fe import apply_cat_triple_cross
        cat_a, cat_b, cat_c = recipe.src_names
        return apply_cat_triple_cross(
            X if (pd is not None and isinstance(X, pd.DataFrame))
            else pd.DataFrame({
                cat_a: _extract_column(X, cat_a),
                cat_b: _extract_column(X, cat_b),
                cat_c: _extract_column(X, cat_c),
            }),
            cat_a, cat_b, cat_c,
            {tuple(k): int(v) for k, v in recipe.extra["mapping"]},
            encoding=str(recipe.extra.get("encoding", "raw")),
            te_lookup={
                int(k): float(v) for k, v in recipe.extra.get("te_lookup", [])
            },
            global_mean=float(recipe.extra.get("global_mean", 0.0)),
        )
    if recipe.kind in ("numeric_rounding", "digit_extract"):
        # Layer 90 (2026-06-01): numeric decomposition (multi-precision
        # rounding + decimal-digit extraction). Pure arithmetic on the single
        # source column -- no lazy import needed, no y reference.
        src_name = recipe.src_names[0]
        vals = _extract_column(X, src_name)
        if recipe.kind == "numeric_rounding":
            from .._numeric_decompose_fe import apply_rounding
            return apply_rounding(vals, float(recipe.extra["precision"]))
        from .._numeric_decompose_fe import apply_digit_extract
        return apply_digit_extract(vals, int(recipe.extra["digit_position"]))
    if recipe.kind == "modular":
        # Layer 95 PART A (2026-06-01): periodic / modular decomposition.
        # Pure arithmetic (x mod period + sin/cos phase) on the single source
        # column -- no lazy import needed, no y reference.
        from .._periodic_fe import apply_modular
        src_name = recipe.src_names[0]
        vals = _extract_column(X, src_name)
        return apply_modular(
            vals, float(recipe.extra["period"]), str(recipe.extra["op"]),
        )
    if recipe.kind == "pairwise_modular":
        # Pairwise / n-way modular residue: combine the source columns (sum/diff/prod/sum3/self) then take mod modulus.
        # Pure integer arithmetic on X, no y reference -> leak-free, train/test exact.
        from .._pairwise_modular_fe import apply_pairwise_modular
        return apply_pairwise_modular(
            X, str(recipe.extra["op"]), recipe.src_names, int(recipe.extra["modulus"]),
        )
    if recipe.kind == "pairwise_integer_lattice":
        # Pairwise integer-lattice column: cast both source columns to int then apply gcd / lcm / bitwise_and.
        # Pure integer arithmetic on X, no y reference -> leak-free, train/test exact.
        from .._integer_lattice_fe import apply_integer_lattice
        return apply_integer_lattice(X, str(recipe.extra["op"]), recipe.src_names)
    if recipe.kind == "group_distance":
        # Layer 95 PART B (2026-06-01): per-group distribution-distance FE
        # (group-level z / KL / Wasserstein-1 from the global distribution).
        # Replay maps a row's group key through the stored per-group scalar
        # lookup; reads only X. Lazy import keeps this module dependency-light.
        from .._group_distance_fe import _apply_group_distance_recipe
        return _apply_group_distance_recipe(recipe, X)
    if recipe.kind == "rare_category":
        # Layer 104 (2026-06-01): rare-category indicator / frequency-band.
        # Replay maps a row's category through the stored per-category frequency
        # lookup; reads only X. Lazy import keeps this module dependency-light.
        from .._extra_fe_families import _apply_rare_category_recipe
        return _apply_rare_category_recipe(recipe, X)
    if recipe.kind == "conditional_residual":
        # Layer 104 (2026-06-01): NUM x NUM conditional residual
        # x_i - E[x_i | bin(x_j)]. Replay digitises x_j with the stored quantile
        # edges and subtracts the stored per-bin mean of x_i; reads only X.
        from .._extra_fe_families import _apply_conditional_residual_recipe
        return _apply_conditional_residual_recipe(recipe, X)
    if recipe.kind == "conditional_dispersion":
        # Family D (backlog #12, 2026-06-09): NUM x NUM conditional DISPERSION /
        # 2nd-moment. Replay digitises x_j with the stored quantile edges, looks
        # up the per-bin (mu_hat, sigma_hat) of x_i, and computes the conditional
        # z-score / |z| / z^2 closed-form; reads only X. Extends Family B's
        # bin_mean payload with bin_std.
        from .._extra_fe_families import _apply_conditional_dispersion_recipe
        return _apply_conditional_dispersion_recipe(recipe, X)
    if recipe.kind == "rankgauss":
        # Layer 104 (2026-06-01): rank-Gaussianisation (RankGauss). Replay
        # interpolates each test value's rank against the stored sorted fit
        # values and maps to a Gaussian quantile; reads only X. Monotone ->
        # MI-invariant by the DPI; value is downstream (linear / NN).
        from .._extra_fe_families import _apply_rankgauss_recipe
        return _apply_rankgauss_recipe(recipe, X)
    if recipe.kind in ("temporal_expanding", "temporal_rolling", "temporal_lag"):
        # Layer 92 (2026-06-01): leak-safe temporal aggregations. Replay
        # computes each test row's expanding / rolling / lag stat against the
        # stored TRAIN per-entity history plus earlier within-test rows -- never
        # the row's own future, never train labels. Lazy import keeps this
        # module under the LOC ceiling.
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
    if recipe.kind in ("grouped_quantile", "target_aware_group_bin"):
        # Layer 88 lazy import: per-group distributional FE (percentile-rank +
        # spread + target-aware supervised bins); replay helpers live with the
        # generator. Keeps this module under the LOC ceiling.
        from .._grouped_quantile_fe import (
            _apply_grouped_quantile_recipe,
            _apply_target_aware_group_bin_recipe,
        )
        if recipe.kind == "grouped_quantile":
            return _apply_grouped_quantile_recipe(recipe, X)
        return _apply_target_aware_group_bin_recipe(recipe, X)
    raise ValueError(f"Unknown recipe kind: {recipe.kind!r}")
