"""Recipe-based replay of engineered features for ``MRMR.transform``.

A *recipe* is a frozen description of how to recompute one engineered column from the original feature matrix. ``MRMR.fit`` records one recipe per surviving
engineered feature; ``MRMR.transform`` replays each recipe against the test X and appends the resulting columns to the output.

Recipe kinds:
``"unary_binary"``: numeric pair FE -- ``binary(unary_a(X[a]), unary_b(X[b]))``, optionally discretized.
``"factorize"``:    cat-FE -- ``merge_vars`` of k ordinal-encoded categorical columns (XOR-style synergy capture).
``"target_encoding"``: ``E[Y | merged_class]`` per cell, with optional OOF smoothing.

The recipe is a small frozen dataclass (no behaviour bound to ``self``) so it round-trips cleanly through pickle and ``sklearn.base.clone``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _extra_equal(a: dict, b: dict) -> bool:
    """Array-aware dict equality for ``EngineeredRecipe.extra``. Plain ``dict.__eq__`` returns an ndarray (not bool) when values are arrays; this helper uses
    ``np.array_equal`` for arrays and ``==`` otherwise.

    2026-05-30 Wave 9.1 fix (loop iter 45): three correctness gaps:
      1. ``np.array_equal(va, vb)`` returns False on NaN-containing
         arrays because NaN != NaN. Persisted recipes whose lookups /
         diagnostics contained NaN (factorize/target_encoding lookups,
         cluster_aggregate's ``pca_var_ratio`` when PCA degenerates)
         failed pickle round-trip equality and ``sklearn.clone`` ==
         fitted checks.
      2. Scalar NaN in the else branch (``va != vb`` is True for
         ``nan != nan``) had the same defect.
      3. Nested list-of-arrays raised ``ValueError: truth value
         ambiguous`` from ``va != vb`` instead of returning bool -
         leaking an exception out of ``__eq__``.
    Fix: NaN-aware array equality via ``equal_nan=True``; NaN-aware
    scalar equality; defensive fallback that returns False on
    ambiguous truth-value errors instead of raising.
    """
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not (isinstance(va, np.ndarray) and isinstance(vb, np.ndarray)):
                return False
            # NaN-aware (equal_nan=True) for float arrays; harmless for
            # integer arrays (numpy ignores the kwarg there).
            try:
                if not np.array_equal(va, vb, equal_nan=True):
                    return False
            except TypeError:
                # Older numpy without equal_nan; fall back to manual
                # NaN-aware check.
                if va.shape != vb.shape:
                    return False
                _eq = (va == vb) | (
                    (va != va) & (vb != vb)  # both NaN -> equal
                )
                if not bool(_eq.all()):
                    return False
        else:
            # Scalar NaN: float('nan') != float('nan') is True under
            # standard comparison, so treat both-NaN as equal.
            if isinstance(va, float) and isinstance(vb, float):
                if (va != va) and (vb != vb):
                    continue
            try:
                if va != vb:
                    return False
            except ValueError:
                # Nested list-of-arrays / ambiguous truth value: be
                # conservative and report unequal rather than raising
                # from inside __eq__.
                return False
    return True


@dataclass(frozen=True, eq=False)
class EngineeredRecipe:
    """One frozen description of how to recompute an engineered column. Survives pickle / ``sklearn.clone`` (no closures or fitted estimators captured).

    Parameters
    ----------
    name
        Engineered column name (e.g. ``"mul(log(c1),sin(c2))"``). Used in transform output / ``get_feature_names_out``.
    kind
        Replay strategy: ``"unary_binary"`` (numeric pair FE), ``"factorize"`` (cat-FE k-way ordinal merge), ``"target_encoding"``.
    src_names
        Original feature names this recipe consumes. Length 2 for unary_binary, k for factorize. Must be a subset of ``feature_names_in_``.
    unary_names
        ``"unary_binary"``: the two unary fn names from ``feature_engineering.create_unary_transformations(preset)``. ``"identity"`` means no transform.
    binary_name
        ``"unary_binary"``: the binary fn name from ``feature_engineering.create_binary_transformations(preset)``.
    unary_preset / binary_preset
        Preset names captured at fit time so later registry edits don't silently change replay semantics.
    quantization
        ``None`` for raw numeric output, else ``{"nbins": int, "method": str, "dtype": str}`` matching fit-time discretization.
    factorize_nbins
        ``"factorize"``: per-source nbins captured at fit time (shape for ``merge_vars`` and bound for ``unknown_strategy`` clipping).
    unknown_strategy
        ``"factorize"`` test-time handling for unseen category values: ``"clip"`` caps at highest trained bin (default); ``"sentinel"`` adds a separate bin
        (inflates cardinality); ``"raise"`` errors out.
    """

    name: str
    # T1#3 2026-05-18 #1 Hermite recipe: ``"hermite_pair"`` kind carries
    # ``coef_a``, ``coef_b``, ``basis``, ``bin_func_name``, ``preprocess_a``,
    # ``preprocess_b``, ``degree_a``, ``degree_b`` in ``extra``. The
    # 88-min Optuna best_res is now reproducible at predict-time.
    # Layer 23 2026-05-31: ``orth_univariate`` carries (src_names=(c,),
    # extra={basis, degree}); ``orth_pair_cross`` carries
    # (src_names=(c_i, c_j), extra={basis_i, basis_j, deg_a, deg_b}). Replay
    # is closed-form from the source column(s) alone -- no y reference is
    # captured at fit time, so transform() is leakage-free by construction.
    kind: Literal["unary_binary", "factorize", "hermite_pair", "target_encoding", "cluster_aggregate", "orth_univariate", "orth_pair_cross", "orth_triplet_cross", "orth_quadruplet_cross", "orth_spline", "orth_fourier", "orth_diff_basis", "orth_cluster_basis", "mi_greedy_transform", "kfold_target_encoded", "count_encoded", "frequency_encoded", "cat_num_residual", "missing_indicator", "missingness_count", "missingness_pattern", "pairwise_ratio", "grouped_delta", "lagged_diff", "grouped_agg", "composite_group_agg", "grouped_quantile", "target_aware_group_bin", "cat_pair_cross", "numeric_rounding", "digit_extract", "temporal_expanding", "temporal_rolling", "temporal_lag", "modular", "group_distance", "rare_category", "conditional_residual", "rankgauss"]
    src_names: tuple[str, ...]
    unary_names: tuple[str, ...] = ()
    binary_name: str = ""
    unary_preset: str = "minimal"
    binary_preset: str = "minimal"
    quantization: dict | None = None
    factorize_nbins: tuple[int, ...] = ()
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip"
    # Free-form bucket for future recipe kinds (e.g. polynomial-basis Hermite carries coef_a/coef_b/degree_a/degree_b/bin_func_name).
    extra: dict = field(default_factory=dict)

    def __getstate__(self):
        """Pickle-friendly state: unwrap the ``MappingProxyType`` back
        to a plain dict so pickle can handle it. ``__setstate__``
        re-applies the post_init freeze.
        """
        state = dict(self.__dict__)
        # Convert mappingproxy -> dict for pickle.
        if "extra" in state:
            state["extra"] = dict(state["extra"])
        return state

    def __setstate__(self, state):
        # ``frozen=True`` blocks normal __dict__ writes; use
        # ``object.__setattr__`` for each key, then re-apply the
        # post_init proxy/freeze chain.
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        """2026-05-30 Wave 9.1 fix (loop iter 49): freeze ``extra``.

        ``frozen=True`` blocks attribute REBIND (``recipe.extra = {}``
        raises) but NOT in-place mutation of the dict itself. Caller-
        held references and accidental ``recipe.extra['x'] = ...`` /
        ``recipe.extra['cell_means'][:] = ...`` silently corrupted
        every subsequent ``apply_recipe`` replay and could poison any
        cache that stored the recipe as a dict/set key (hash stays the
        same on ``(kind, name)`` while ``__eq__`` flips with content).

        Four failure modes documented in the iter-49 repro:
        H.1 caller pops a required key after construction -> apply_*
            raises KeyError on what looked like a "frozen" recipe.
        H.2 in-place ndarray mutation -> apply_* returns garbage.
        H.3 hash-eq invariant violated for any recipe in a set/dict.
        H.4 cache poisoning when recipe used as dict key.

        Fix: deep-copy the ``extra`` dict at construction (severs the
        caller-held reference), freeze every ndarray inside it, then
        wrap in ``MappingProxyType`` (read-only view). Re-assigning via
        ``object.__setattr__`` because ``frozen=True`` blocks normal
        attribute writes inside ``__post_init__``.
        """
        import copy as _copy_iter49
        import types as _types_iter49
        # Deep-copy so post-construction mutation of caller's source
        # dict can't propagate into the recipe.
        _extra_copy = _copy_iter49.deepcopy(dict(self.extra))
        # Freeze every ndarray value so ``recipe.extra['x'][:] = ...``
        # raises ValueError instead of silently corrupting downstream
        # replays. Skip arrays we don't own (views).
        for _v in _extra_copy.values():
            if isinstance(_v, np.ndarray):
                if _v.flags.owndata and _v.flags.writeable:
                    try:
                        _v.flags.writeable = False
                    except Exception:
                        pass
        # Wrap in read-only proxy. ``MappingProxyType`` returns
        # ``TypeError`` on any ``extra['x'] = ...`` style write.
        object.__setattr__(
            self, "extra", _types_iter49.MappingProxyType(_extra_copy),
        )

    def __eq__(self, other: object) -> bool:
        """Custom ``__eq__`` handling ndarray values in ``extra`` (factorize lookup tables). ``frozen=True, eq=False`` disables the auto-generated one."""
        if not isinstance(other, EngineeredRecipe):
            return NotImplemented
        if self.kind != other.kind:
            return False
        if self.name != other.name:
            return False
        if self.src_names != other.src_names:
            return False
        if self.unary_names != other.unary_names:
            return False
        if self.binary_name != other.binary_name:
            return False
        if self.unary_preset != other.unary_preset:
            return False
        if self.binary_preset != other.binary_preset:
            return False
        if self.quantization != other.quantization:
            return False
        if self.factorize_nbins != other.factorize_nbins:
            return False
        if self.unknown_strategy != other.unknown_strategy:
            return False
        return _extra_equal(self.extra, other.extra)

    def __hash__(self) -> int:
        # Name-based hash (names are unique per fit), since ``extra: dict`` is mutable and would normally disable __hash__.
        # Wave 73 (2026-05-21) hardening: __eq__ (above) walks the ``extra`` dict
        # content (incl. ndarrays via np.array_equal). Hash key (kind, name) is
        # NARROWER than equality, so two recipes with same (kind, name) but
        # different ``extra`` collide on the same hash bucket but DON'T compare
        # equal. That's a valid hash-eq pair (equal-implies-equal-hash holds),
        # but it WOULD trigger an O(N) bucket scan if recipes were ever stored
        # in a set/dict-key with name collisions on different content.
        # Contract: callers MUST NOT use ``EngineeredRecipe`` instances as
        # dict/set keys when the same ``name`` can carry different ``extra``;
        # use ``recipe.name`` (the string) as the dict key instead. All current
        # callers store recipes as dict VALUES (engineered_recipes[r.name] = r).
        return hash((self.kind, self.name))


def _extract_column(X: Any, name: str) -> np.ndarray:
    """Pull a single column from X by name as a 1-D ndarray, no full-frame copy. Supports pandas / polars DataFrame and numpy structured arrays."""
    if pd is not None and isinstance(X, pd.DataFrame):
        # ``.values`` is zero-copy for numeric dtypes; categorical/object materialises only the single column.
        return X[name].to_numpy() if hasattr(X[name], "to_numpy") else X[name].values
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[name].to_numpy()
    if isinstance(X, np.ndarray):
        if X.dtype.names is not None:
            return X[name]
        raise KeyError(
            f"Cannot resolve column '{name}' on a plain 2-D ndarray. "
            "Pass a pandas / polars frame or a structured array."
        )
    raise TypeError(f"Unsupported X type for engineered-recipe replay: {type(X)!r}")


def _coerce_to_int_with_nan_handling(
    vals: np.ndarray, n_bins: int, recipe_name: str, col_name: str,
    unknown_strategy: str,
) -> np.ndarray:
    """Coerce test-time values to int64 for factorize lookup, handling NaN/non-integer per ``unknown_strategy`` (clip -> max bin, sentinel -> new bin,
    raise -> error). Float non-NaN casts to int (rounds toward zero); object/categorical via ``astype(int64)``."""
    if np.issubdtype(vals.dtype, np.floating):
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            if unknown_strategy == "raise":
                n_nan = int(nan_mask.sum())
                raise ValueError(
                    f"Recipe '{recipe_name}': column '{col_name}' has "
                    f"{n_nan} NaN value(s) at transform time. Set "
                    f"unknown_strategy='clip' or 'sentinel' to handle "
                    f"silently."
                )
            # 'clip'/'sentinel' both leave NaN unhandled at the float level; replace with n_bins-1 sentinel so the lookup (which encoded the strategy at
            # fit time) resolves it. The clip path in ``_apply_factorize`` handles the rest.
            vals = vals.copy()
            vals[nan_mask] = n_bins - 1
        return vals.astype(np.int64, copy=False)
    if np.issubdtype(vals.dtype, np.integer):
        return vals.astype(np.int64, copy=False)
    # Object / categorical / string -- try int conversion
    try:
        return vals.astype(np.int64, copy=False)
    except (ValueError, TypeError) as e:
        if unknown_strategy == "raise":
            raise ValueError(
                f"Recipe '{recipe_name}': column '{col_name}' has "
                f"non-integer dtype {vals.dtype!r} that cannot be "
                f"coerced. Pass ordinal-encoded ints or set "
                f"unknown_strategy='clip'."
            ) from e
        # Clip-equivalent fallback: all to 0
        return np.zeros(len(vals), dtype=np.int64)


def apply_recipe(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay ``recipe`` against ``X`` and return the engineered column as a 1-D ndarray. Output dtype matches the recipe's quantization dtype if
    discretized, else float32 (matching fit-time ``check_prospective_fe_pairs`` working buffer dtype). Hot path in ``transform()`` -- keep allocation-light."""
    if recipe.kind == "unary_binary":
        return _apply_unary_binary(recipe, X)
    if recipe.kind == "factorize":
        return _apply_factorize(recipe, X)
    if recipe.kind == "target_encoding":
        return _apply_target_encoding(recipe, X)
    if recipe.kind == "hermite_pair":
        return _apply_hermite_pair(recipe, X)
    if recipe.kind == "cluster_aggregate":
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


def _apply_hermite_pair(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """T1#3 2026-05-18: replay a Hermite/Chebyshev/Laguerre polynomial-pair FE column at predict time.

    Carries the full ``HermiteResult`` state (coefficients, basis name,
    bin-function name, preprocessing parameters) in ``recipe.extra``;
    the builder ``build_hermite_pair_recipe`` populates it. Lazy import
    of ``hermite_fe`` keeps this module dependency-light at import time.
    """
    if len(recipe.src_names) != 2:
        raise ValueError(
            f"hermite_pair recipe '{recipe.name}' must have exactly 2 src_names; "
            f"got {len(recipe.src_names)}"
        )
    for key in ("coef_a", "coef_b", "basis", "bin_func_name",
                "preprocess_a", "preprocess_b"):
        if key not in recipe.extra:
            raise KeyError(
                f"hermite_pair recipe '{recipe.name}' missing '{key}' in extra. "
                f"Re-fit with the current build_hermite_pair_recipe to repopulate."
            )

    # Lazy imports to avoid circular dependency (hermite_fe -> mrmr -> recipes).
    from ..hermite_fe import _POLY_BASES, _DEFAULT_BIN_FUNCS

    basis = recipe.extra["basis"]
    bin_func_name = recipe.extra["bin_func_name"]
    if basis not in _POLY_BASES:
        raise KeyError(
            f"hermite_pair recipe '{recipe.name}' references unknown basis "
            f"{basis!r}; available: {sorted(_POLY_BASES)}"
        )
    if bin_func_name not in _DEFAULT_BIN_FUNCS:
        raise KeyError(
            f"hermite_pair recipe '{recipe.name}' references unknown "
            f"bin_func_name {bin_func_name!r}; available: "
            f"{sorted(_DEFAULT_BIN_FUNCS)}"
        )

    name_a, name_b = recipe.src_names
    try:
        vals_a = _extract_column(X, name_a)
        vals_b = _extract_column(X, name_b)
    except Exception as _src_err:
        # A source can be a fit-time-pruned engineered intermediate that is unreconstructable at replay. The mrmr
        # validate-transform contract NaN-degrades chained-capable kinds (hermite_pair references engineered cols)
        # rather than crash the whole transform; mirror that here for the same missing-source class, re-raise else.
        if isinstance(_src_err, (KeyError, IndexError)) or "ColumnNotFound" in type(_src_err).__name__:
            return np.full(len(X), np.nan, dtype=np.float64)
        raise

    basis_info = _POLY_BASES[basis]
    z_a = np.ascontiguousarray(
        basis_info["apply"](vals_a, dict(recipe.extra["preprocess_a"])),
        dtype=np.float64,
    )
    z_b = np.ascontiguousarray(
        basis_info["apply"](vals_b, dict(recipe.extra["preprocess_b"])),
        dtype=np.float64,
    )
    eval_dispatch = basis_info["eval_dispatch"]
    coef_a = np.ascontiguousarray(recipe.extra["coef_a"], dtype=np.float64)
    coef_b = np.ascontiguousarray(recipe.extra["coef_b"], dtype=np.float64)
    h_a = eval_dispatch(z_a, coef_a)
    h_b = eval_dispatch(z_b, coef_b)
    bin_func = _DEFAULT_BIN_FUNCS[bin_func_name]
    return np.asarray(bin_func(h_a, h_b), dtype=np.float64).reshape(-1)


def build_hermite_pair_recipe(
    *, name: str, src_names: tuple[str, str], hermite_result,
) -> EngineeredRecipe:
    """T1#3 2026-05-18: builder that turns a ``HermiteResult`` (output of ``optimise_hermite_pair`` / Optuna driver) into a frozen ``EngineeredRecipe`` survivable across pickle / sklearn.clone.

    The recipe captures the coefficients, basis name, bin-function name,
    and preprocessing parameters - exactly the state required for
    ``_apply_hermite_pair`` to reproduce the column on a test frame.
    """
    return EngineeredRecipe(
        name=name,
        kind="hermite_pair",
        src_names=src_names,
        extra={
            "coef_a": np.asarray(hermite_result.coef_a, dtype=np.float64).copy(),
            "coef_b": np.asarray(hermite_result.coef_b, dtype=np.float64).copy(),
            "basis": str(hermite_result.basis),
            "bin_func_name": str(hermite_result.bin_func_name),
            "preprocess_a": dict(hermite_result.preprocess_a or {}),
            "preprocess_b": dict(hermite_result.preprocess_b or {}),
            "degree_a": int(hermite_result.degree_a),
            "degree_b": int(hermite_result.degree_b),
        },
    )


def build_cluster_aggregate_recipe(
    *, name: str, src_names: tuple[str, ...], method: str,
    member_mean: np.ndarray, member_std: np.ndarray, signs: np.ndarray,
    weights: np.ndarray | None = None, quantization: dict | None = None,
    diagnostics: dict | None = None,
) -> EngineeredRecipe:
    """Frozen recipe for a k-ary cluster aggregate (denoised cluster representative).

    Replay (``_apply_cluster_aggregate``) standardizes each member with the STORED train ``member_mean``
    / ``member_std``, sign-aligns with ``signs``, then for every linear combiner forms ``Z @ weights``
    (``mean_z`` -> uniform 1/k, ``mean_inv_var`` -> normalized 1/sigma_hat^2, ``pca_pc1`` -> PC1
    eigenvector, ``factor_score`` -> Bartlett combiner). ``median`` ignores ``weights``. All ``extra``
    values are flat ndarray / scalar / str (``_extra_equal`` is not deep)."""
    extra: dict = {
        "method": str(method),
        "member_mean": np.asarray(member_mean, dtype=np.float64).copy(),
        "member_std": np.asarray(member_std, dtype=np.float64).copy(),
        "signs": np.asarray(signs, dtype=np.float64).copy(),
    }
    if weights is not None:
        extra["weights"] = np.asarray(weights, dtype=np.float64).copy()
    for k, v in (diagnostics or {}).items():  # e.g. pca_var_ratio, representative -- scalars/str only
        extra[k] = v
    # Build-time guard: _extra_equal compares values shallowly (np.array_equal / !=), so nested
    # lists/dicts of arrays would break __eq__/pickle round-trip. Keep extra flat.
    for k, v in extra.items():
        assert isinstance(v, (np.ndarray, str, int, float, bool)), f"cluster_aggregate extra[{k!r}] must be flat ndarray/scalar/str, got {type(v)}"
    return EngineeredRecipe(name=name, kind="cluster_aggregate", src_names=tuple(src_names), quantization=quantization, extra=extra)


def _apply_cluster_aggregate(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a cluster-aggregate column: standardize members with stored train stats, sign-align,
    combine (``Z @ weights`` for linear methods, per-row median for ``median``), then discretize.

    Stateless given the stored ``extra`` -> uses ONLY train-fitted stats (never re-standardizes on the
    test distribution), so train/test parity holds. Pure numpy -> no lazy import / import-cycle risk."""
    for key in ("method", "member_mean", "member_std", "signs"):
        if key not in recipe.extra:
            raise KeyError(f"cluster_aggregate recipe '{recipe.name}' missing '{key}' in extra. Re-fit to repopulate.")
    method = recipe.extra["method"]
    member_mean = np.asarray(recipe.extra["member_mean"], dtype=np.float64)
    member_std = np.asarray(recipe.extra["member_std"], dtype=np.float64)
    signs = np.asarray(recipe.extra["signs"], dtype=np.float64)
    if not (len(recipe.src_names) == len(member_mean) == len(member_std) == len(signs)):
        raise ValueError(f"cluster_aggregate recipe '{recipe.name}': src_names / stats length mismatch.")

    # 2026-05-30 Wave 9.1 fix (loop iter 8): match the train-time
    # ``_continuous_cols`` preprocessing in ``_cluster_aggregate.py:119``
    # which ``nan_to_num``s the member columns BEFORE computing mean / std.
    # Without the same wrap here, replay rows with NaN in any member become
    # ``(NaN - mean) / std = NaN``, the ``Z @ weights`` row becomes NaN, and
    # the final ``nan_to_num(out)`` at line 352 zeroes the row -- but the
    # train-time aggregate for the same row was ``((0 - mean) / std * sign)
    # @ weights``, a specific value. So fit recorded one number and
    # transform produced a different one, breaking train/test parity in
    # the very common case where TRAIN itself contains NaN.
    cols = [
        np.nan_to_num(
            np.asarray(_extract_column(X, n), dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        for n in recipe.src_names
    ]
    M = np.column_stack(cols)  # (n, k)
    std = np.where(member_std > 0.0, member_std, 1.0)
    Z = ((M - member_mean) / std) * signs  # standardize with TRAIN stats + sign-align

    # Layer 44: dispatch the four new non-linear / row-reduction methods that
    # have no weights vector. Replay logic mirrors ``_apply_method_nonlinear``
    # at fit; kept inline here so engineered_recipes.py stays free of an import
    # cycle with _cluster_aggregate.py.
    if method in ("median", "median_z"):
        out = np.median(Z, axis=1)
    elif method == "signed_max_abs":
        abs_Z = np.abs(Z)
        idx = np.argmax(abs_Z, axis=1)
        rows = np.arange(Z.shape[0])
        signs_row = np.sign(Z[rows, idx])
        signs_row = np.where(signs_row == 0.0, 1.0, signs_row)
        out = signs_row * abs_Z[rows, idx]
    elif method == "signed_l2_sum":
        out = np.sum(np.sign(Z) * (Z ** 2), axis=1)
    else:
        if "weights" not in recipe.extra:
            raise KeyError(f"cluster_aggregate recipe '{recipe.name}' (method={method!r}) missing 'weights' in extra.")
        out = Z @ np.asarray(recipe.extra["weights"], dtype=np.float64)

    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if recipe.quantization is not None:
        from ..discretization import discretize_array
        q = recipe.quantization
        # 2026-05-30 Wave 9.1 fix (loop iter 29): use fit-time edges when
        # the recipe stored them. Pre-fix ``discretize_array`` recomputed
        # ``np.nanpercentile`` from TEST aggregate values each replay, so
        # the same physical input row mapped to DIFFERENT cluster_aggregate
        # bin codes between fit and transform under any distribution drift
        # (83% disagreement at 10x stddev shift). Sibling of iter 28's
        # unary_binary fix.
        if q.get("edges") is not None:
            edges = np.asarray(q["edges"], dtype=np.float64)
            out = np.searchsorted(
                edges[1:-1] if edges.size >= 2 else edges,
                out, side="right",
            ).astype(np.dtype(q["dtype"]))
        else:
            import warnings as _w_iter29
            _w_iter29.warn(
                f"cluster_aggregate recipe '{recipe.name}' has no fit-time "
                f"quantile edges; replay will re-quantile on test data and "
                f"produce shifted codes under distribution drift. Refit the "
                f"MRMR estimator to regenerate the recipe with persisted edges.",
                UserWarning, stacklevel=2,
            )
            out = discretize_array(arr=out, n_bins=q["nbins"], method=q["method"], dtype=np.dtype(q["dtype"]))
    return out


def _apply_target_encoding(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Look up each test row's (a, b) merged cell in ``cell_means``, return per-row encoded float. Unseen combinations map to ``global_mean`` (or per ``unknown_strategy``)."""
    if len(recipe.src_names) != 2:
        raise NotImplementedError(
            f"target_encoding for k>2 not implemented yet "
            f"(recipe '{recipe.name}' has {len(recipe.src_names)} src)."
        )
    if "cell_means" not in recipe.extra or "factorize_lookup" not in recipe.extra:
        raise KeyError(
            f"target_encoding recipe '{recipe.name}' is missing 'cell_means' "
            f"or 'factorize_lookup' in extra. Re-fit to materialize."
        )
    name_a, name_b = recipe.src_names
    nbins_a, nbins_b = recipe.factorize_nbins
    factorize_lookup: np.ndarray = recipe.extra["factorize_lookup"]
    cell_means: np.ndarray = recipe.extra["cell_means"]
    global_mean: float = recipe.extra["global_mean"]

    vals_a = _coerce_to_int_with_nan_handling(
        _extract_column(X, name_a), nbins_a, recipe.name, name_a, recipe.unknown_strategy,
    )
    vals_b = _coerce_to_int_with_nan_handling(
        _extract_column(X, name_b), nbins_b, recipe.name, name_b, recipe.unknown_strategy,
    )
    vals_a = np.clip(vals_a, 0, nbins_a - 1)
    vals_b = np.clip(vals_b, 0, nbins_b - 1)
    pre_prune = vals_a + vals_b * nbins_a
    cell_idx = factorize_lookup[pre_prune]
    # 2026-05-30 Wave 9.1 fix (loop iter 22): honour ``unknown_strategy='raise'``.
    # Pre-fix this branch silently substituted ``global_mean`` for unseen
    # cells even when the recipe was explicitly built with raise strategy,
    # diverging from ``_apply_factorize`` (line 512) and
    # ``_apply_factorize_kway`` (line 556, iter-13 fix) which both raise.
    # Production monitoring relying on raise-on-drift was failing open:
    # a model whose target-encoded features were silently filled with the
    # train global mean degraded predictions but passed any try/except
    # guard.
    if recipe.unknown_strategy == "raise" and (cell_idx < 0).any():
        n_unseen = int((cell_idx < 0).sum())
        raise ValueError(
            f"target_encoding recipe '{recipe.name}': {n_unseen} row(s) have "
            f"(X[{name_a}], X[{name_b}]) combinations not seen during fit. "
            f"Set unknown_strategy='clip' or 'sentinel' to handle silently."
        )
    out = np.where(cell_idx >= 0, cell_means[np.maximum(cell_idx, 0)], global_mean)
    return out.astype(np.float64, copy=False)


def _apply_unary_binary(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        raise ValueError(
            f"unary_binary recipe '{recipe.name}' must have exactly 2 src_names "
            f"and 2 unary_names; got {len(recipe.src_names)} / {len(recipe.unary_names)}"
        )
    # Lazy import to avoid circular dependency (feature_engineering -> mrmr via _internals).
    from ..feature_engineering import create_unary_transformations, create_binary_transformations
    from ..discretization import discretize_array

    unary_funcs = create_unary_transformations(preset=recipe.unary_preset)
    binary_funcs = create_binary_transformations(preset=recipe.binary_preset)

    name_a, name_b = recipe.src_names
    u_a, u_b = recipe.unary_names

    # The learned ``prewarp`` (2026-06-02) and ``gate_med`` (2026-06-04)
    # pseudo-unaries are NOT members of any preset registry; their closed-form
    # replay reads the fit-time state stored in ``recipe.extra`` (prewarp: poly
    # coeffs; gate_med: the TRAIN median). Validate only the REAL unary names
    # against the preset.
    _PREWARP = "prewarp"
    _GATE_MED = "gate_med"
    _PSEUDO = (_PREWARP, _GATE_MED)
    if u_a not in _PSEUDO and u_a not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_a}' not in '{recipe.unary_preset}' preset. "
            f"Replay requires the same preset that was active at fit time."
        )
    if u_b not in _PSEUDO and u_b not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_b}' not in '{recipe.unary_preset}' preset."
        )
    if recipe.binary_name not in binary_funcs:
        raise KeyError(
            f"Binary function '{recipe.binary_name}' not in "
            f"'{recipe.binary_preset}' preset."
        )

    # NESTED-ENGINEERED PARENTS (2026-06-08): when an operand is itself an engineered
    # column (a higher-order composite), its values are NOT in ``X`` at transform time
    # (X carries only raw columns). Recompute it by recursively replaying the stored
    # parent recipe, forced to its CONTINUOUS output (quantization stripped) so the
    # composite is built on continuous values exactly as at fit time -- the fit-time
    # operand was the parent's continuous engineered value, not its bin codes.
    def _nested_continuous(parent: "EngineeredRecipe") -> np.ndarray:
        if parent.quantization is not None:
            # Replay the parent WITHOUT quantization (continuous output).
            import dataclasses as _dc
            parent = _dc.replace(parent, quantization=None)
        return np.asarray(apply_recipe(parent, X), dtype=np.float64)

    _np_a = recipe.extra.get("nested_parent_a")
    _np_b = recipe.extra.get("nested_parent_b")
    vals_a = _nested_continuous(_np_a) if _np_a is not None else _extract_column(X, name_a)
    vals_b = _nested_continuous(_np_b) if _np_b is not None else _extract_column(X, name_b)

    def _apply_side(side: str, uname: str, vals):
        if uname == _GATE_MED:
            # Median-gate pseudo-unary: replay closed-form ``(x > train_median)``
            # from the single fit-time median float stored in ``extra`` (no y, no
            # test-time recompute -> leak-safe and bit-identical to fit).
            _mkey = f"gate_med_{side}_median"
            if _mkey not in recipe.extra:
                raise KeyError(
                    f"unary_binary recipe '{recipe.name}' uses the 'gate_med' "
                    f"pseudo-unary on side {side!r} but '{_mkey}' is missing from "
                    f"extra. Re-fit MRMR to regenerate the recipe."
                )
            from .._feature_engineering_pairs import _gate_med_apply
            return _gate_med_apply(vals, float(recipe.extra[_mkey]))
        if uname != _PREWARP:
            return unary_funcs[uname](vals)
        # Reconstruct the pre-warp spec from the flat ``extra`` fields and replay
        # it closed-form (no y) so the warped operand is bit-identical to fit.
        import orjson as _orjson
        from ..hermite_fe import apply_operand_prewarp
        for _k in (f"prewarp_{side}_coef", f"prewarp_{side}_basis"):
            if _k not in recipe.extra:
                raise KeyError(
                    f"unary_binary recipe '{recipe.name}' uses the 'prewarp' "
                    f"pseudo-unary on side {side!r} but '{_k}' is missing from "
                    f"extra. Re-fit MRMR to regenerate the recipe."
                )
        _spec = {
            "basis": str(recipe.extra[f"prewarp_{side}_basis"]),
            "degree": int(recipe.extra[f"prewarp_{side}_degree"]),
            "coef": np.asarray(recipe.extra[f"prewarp_{side}_coef"], dtype=np.float64),
            "preprocess": _orjson.loads(recipe.extra[f"prewarp_{side}_preprocess"]),
        }
        return apply_operand_prewarp(vals, _spec)

    transformed_a = _apply_side("a", u_a, vals_a)
    transformed_b = _apply_side("b", u_b, vals_b)
    out = binary_funcs[recipe.binary_name](transformed_a, transformed_b)

    # Match fit-time NaN/Inf scrubbing in ``feature_engineering.check_prospective_fe_pairs``.
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if recipe.quantization is not None:
        q = recipe.quantization
        # 2026-05-30 Wave 9.1 fix (loop iter 28): use fit-time edges when
        # the recipe stored them. Pre-fix ``discretize_array`` recomputed
        # ``np.nanpercentile`` from TEST data each replay, so the same
        # physical row mapped to DIFFERENT bin codes between fit and
        # transform under any distribution drift (58.8% disagreement
        # observed in synthetic shift demo). That's a textbook train/test
        # leak: the model trained on stale bin codes and got fresh
        # rebinned codes at inference.
        if q.get("edges") is not None:
            edges = np.asarray(q["edges"], dtype=np.float64)
            out = np.searchsorted(
                edges[1:-1] if edges.size >= 2 else edges,
                out, side="right",
            ).astype(np.dtype(q["dtype"]))
        else:
            # Back-compat: pre-iter-28 recipes (pickled before edges were
            # persisted) fall back to the leaky path. Warn so maintainers
            # see the smell. New recipes always carry edges.
            import warnings as _w_iter28
            _w_iter28.warn(
                f"unary_binary recipe '{recipe.name}' has no fit-time "
                f"quantile edges; replay will re-quantile on test data "
                f"and produce shifted codes under distribution drift. "
                f"Refit the MRMR estimator to regenerate the recipe with "
                f"persisted edges.",
                UserWarning, stacklevel=2,
            )
            out = discretize_array(
                arr=out,
                n_bins=q["nbins"],
                method=q["method"],
                dtype=np.dtype(q["dtype"]),
            )
    return out


def _apply_factorize(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Cat-FE replay: look up each test row's ``(a, b)`` tuple (or k-way chain) in the fit-time lookup table(s) and emit the post-prune class.

    Pairs (k=2): single lookup maps ``a_value + b_value * nbins_a`` to post-prune class. K > 2: chained lookup via ``recipe.extra['chain_lookups']`` (a list
    of k-1 pair lookups); each step combines running intermediate class with next column's value via ``chain_nuniqs`` from previous step.

    Test values outside ``[0, nbins_i)`` are clipped to ``nbins_i - 1``. Combinations whose pre-prune code never appeared in training are resolved per
    ``recipe.unknown_strategy`` (already baked into each lookup at fit time, except for ``"raise"`` which keeps -1 sentinels and surfaces here).
    """
    if recipe.extra.get("chain_lookups"):
        return _apply_factorize_kway(recipe, X)

    # Defensive branch for old pickled k-way recipes that lack chained lookups.
    if recipe.extra.get("requires_refit_for_replay"):
        raise NotImplementedError(
            f"factorize recipe '{recipe.name}' is a legacy k-way recipe "
            f"(order {recipe.extra.get('kway_order', '?')}) lacking a chained-lookup payload. "
            "Re-fit MRMR to materialise the chained-lookup version for replay."
        )
    if len(recipe.src_names) != 2 or len(recipe.factorize_nbins) != 2:
        raise ValueError(
            f"factorize recipe '{recipe.name}' requires exactly 2 src_names "
            f"and 2 factorize_nbins; got {len(recipe.src_names)} / "
            f"{len(recipe.factorize_nbins)}"
        )
    if "lookup_table" not in recipe.extra:
        raise KeyError(
            f"factorize recipe '{recipe.name}' is missing the 'lookup_table' "
            f"in recipe.extra. This usually means the recipe was built before "
            f"the cat-FE replay PR landed; refit the MRMR estimator."
        )

    name_a, name_b = recipe.src_names
    nbins_a, nbins_b = recipe.factorize_nbins
    lookup: np.ndarray = recipe.extra["lookup_table"]

    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)
    # Handle NaN / non-integer values per unknown_strategy.
    vals_a_i = _coerce_to_int_with_nan_handling(vals_a, nbins_a, recipe.name, name_a, recipe.unknown_strategy)
    vals_b_i = _coerce_to_int_with_nan_handling(vals_b, nbins_b, recipe.name, name_b, recipe.unknown_strategy)

    # Clip out-of-range to nbins-1. Without this, a test value of ``nbins_a + 1`` would index past the lookup buffer end. Per ``unknown_strategy="clip"``
    # semantics (default), unseen values map to the highest seen class -- already encoded in the lookup; here we just guard the buffer.
    vals_a_i = np.clip(vals_a_i, 0, nbins_a - 1)
    vals_b_i = np.clip(vals_b_i, 0, nbins_b - 1)

    pre_prune_codes = vals_a_i + vals_b_i * nbins_a
    out = lookup[pre_prune_codes]

    # ``raise`` strategy left -1 sentinels in the lookup -- anything negative here is a test combo never seen in training.
    if recipe.unknown_strategy == "raise" and (out < 0).any():
        n_unseen = int((out < 0).sum())
        raise ValueError(
            f"factorize recipe '{recipe.name}': {n_unseen} row(s) have "
            f"(X[{name_a}], X[{name_b}]) combinations not seen during fit. "
            f"Set unknown_strategy='clip' or 'sentinel' to handle these "
            f"silently."
        )
    return out


def _apply_factorize_kway(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """K-way replay via the chained-lookup payload. Each ``chain_lookups[step]`` is a flat int64 table indexed by ``running_intermediate + col_value *
    running_nuniq``. We walk through all (k-1) steps, refreshing ``running_intermediate`` and ``running_nuniq`` from each step's output.

    Per-column test values are clipped to ``[0, factorize_nbins[i])``. Unseen combinations resolve per ``recipe.unknown_strategy`` (already encoded at fit
    time, except ``"raise"`` which leaves -1 sentinels and surfaces here).
    """
    src_names = recipe.src_names
    nbins_tuple = recipe.factorize_nbins
    chain_lookups: list = recipe.extra["chain_lookups"]
    chain_nuniqs: list = recipe.extra["chain_nuniqs"]
    k = len(src_names)
    if len(chain_lookups) != k - 1 or len(chain_nuniqs) != k - 1:
        raise ValueError(
            f"k-way recipe '{recipe.name}' chain payload size mismatch: "
            f"k={k}, chain_lookups={len(chain_lookups)}, "
            f"chain_nuniqs={len(chain_nuniqs)} (expected {k-1} each)."
        )

    # Step 1: build running from first two columns
    vals_0 = _coerce_to_int_with_nan_handling(
        _extract_column(X, src_names[0]), int(nbins_tuple[0]),
        recipe.name, src_names[0], recipe.unknown_strategy,
    )
    vals_1 = _coerce_to_int_with_nan_handling(
        _extract_column(X, src_names[1]), int(nbins_tuple[1]),
        recipe.name, src_names[1], recipe.unknown_strategy,
    )
    vals_0 = np.clip(vals_0, 0, int(nbins_tuple[0]) - 1)
    vals_1 = np.clip(vals_1, 0, int(nbins_tuple[1]) - 1)
    pre_prune = vals_0 + vals_1 * int(nbins_tuple[0])
    running = chain_lookups[0][pre_prune]
    running_nuniq = chain_nuniqs[0]
    # 2026-05-30 Wave 9.1 fix (loop iter 13): under unknown_strategy='raise'
    # we MUST raise BEFORE the next chain step uses ``running`` as an index.
    # If we don't, ``pre_prune = running + vals_next * running_nuniq`` with
    # ``running[i] == -1`` (unseen prefix) computes a negative-or-small
    # index that Python wraps to the tail of ``chain_lookups[step-1]``,
    # silently returning a real class code. The post-loop ``(running < 0)``
    # check then sees no -1 and fails to raise. Confirmed live: a 3-way
    # recipe with raise mode + unseen prefix returned [1] instead of
    # raising ValueError.
    if recipe.unknown_strategy == "raise" and (running < 0).any():
        n_unseen = int((running < 0).sum())
        raise ValueError(
            f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
            f"hit unseen prefix at chain step 1. Set unknown_strategy="
            f"'clip' or 'sentinel' to handle silently."
        )

    # Steps 2..k-1: chain forward
    for step in range(2, k):
        vals_next = _coerce_to_int_with_nan_handling(
            _extract_column(X, src_names[step]), int(nbins_tuple[step]),
            recipe.name, src_names[step], recipe.unknown_strategy,
        )
        vals_next = np.clip(vals_next, 0, int(nbins_tuple[step]) - 1)
        pre_prune = running + vals_next * running_nuniq
        running = chain_lookups[step - 1][pre_prune]
        running_nuniq = chain_nuniqs[step - 1]
        # Same guard at every intermediate step: any -1 here would get
        # silently overwritten by the next negative-index wrap.
        if recipe.unknown_strategy == "raise" and (running < 0).any():
            n_unseen = int((running < 0).sum())
            raise ValueError(
                f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
                f"hit unseen prefix at chain step {step}. Set "
                f"unknown_strategy='clip' or 'sentinel' to handle silently."
            )

    if recipe.unknown_strategy == "raise" and (running < 0).any():
        n_unseen = int((running < 0).sum())
        raise ValueError(
            f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
            f"have combinations not seen during fit. Set "
            f"unknown_strategy='clip' or 'sentinel' to handle silently."
        )
    return running


def build_unary_binary_recipe(
    *,
    name: str,
    src_a_name: str,
    src_b_name: str,
    unary_a_name: str,
    unary_b_name: str,
    binary_name: str,
    unary_preset: str,
    binary_preset: str,
    quantization_nbins: int | None,
    quantization_method: str | None,
    quantization_dtype: Any,
    fit_values_for_edges: np.ndarray | None = None,
    prewarp_a: dict | None = None,
    prewarp_b: dict | None = None,
    gate_med_a: float | None = None,
    gate_med_b: float | None = None,
    nested_parent_a: "EngineeredRecipe | None" = None,
    nested_parent_b: "EngineeredRecipe | None" = None,
) -> EngineeredRecipe:
    """Build an ``EngineeredRecipe`` of kind ``"unary_binary"``. ``quantization`` is ``None`` if no discretization, else a dict carrying the binning
    parameters AND, when ``fit_values_for_edges`` is provided, the fit-time bin edges so transform-time replay maps each row to the SAME bin
    code regardless of test-data distribution. Dtype is stringified so the recipe is JSON-friendly and pickle-safe across numpy versions.

    2026-05-30 Wave 9.1 iter 28: ``fit_values_for_edges`` lets the caller pin the quantile boundaries. Without it (legacy code paths) the
    recipe emits a UserWarning at replay time about the train/test leakage risk.

    NESTED-ENGINEERED PARENTS (2026-06-08): ``nested_parent_a`` / ``nested_parent_b`` are
    the ``EngineeredRecipe`` objects for operands that are THEMSELVES engineered columns
    (a step-k>1 composite of two prior engineered features, e.g.
    ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))``). When supplied, the parent recipe is
    stored in ``extra["nested_parent_<side>"]`` and ``_apply_unary_binary`` recomputes that
    operand at transform time by recursively replaying the parent (forced to its CONTINUOUS
    output -- quantization stripped -- so the composite is built on continuous values exactly
    as at fit time) rather than reading ``src_<side>_name`` from ``X`` (which only carries
    raw columns at transform time). This makes higher-order composites fully replayable."""
    # Pre-warp recipes emit a CONTINUOUS learned-polynomial product (the same
    # nature as the orthogonal-poly ``hermite_pair`` recipe, which is NOT
    # quantised at replay). Quantile-binning the heavy-tailed product to integer
    # codes throws away most of its correlation with the target (measured: raw
    # mul-product corr 0.88 vs binned 0.62 on F-POLY). Mirror the hermite_pair
    # sibling and skip quantization so ``transform()`` returns the raw feature;
    # the downstream MRMR fit still discretises the fit-time column for its own
    # MI matrix via ``_mrmr_fe_step`` (unaffected by this replay-only choice).
    _is_prewarp = prewarp_a is not None or prewarp_b is not None
    if quantization_nbins is None or _is_prewarp:
        quantization = None
    else:
        quantization = {
            "nbins": int(quantization_nbins),
            "method": str(quantization_method) if quantization_method else "uniform",
            "dtype": np.dtype(quantization_dtype).str,
        }
        # Persist fit-time edges so replay never re-quantiles on test data.
        if fit_values_for_edges is not None:
            _arr = np.asarray(fit_values_for_edges, dtype=np.float64).ravel()
            if quantization["method"] == "quantile":
                _q = np.linspace(0.0, 100.0, int(quantization_nbins) + 1)
                _edges = np.nanpercentile(_arr, _q)
            else:  # uniform
                _finite = _arr[np.isfinite(_arr)]
                if _finite.size:
                    _lo = float(_finite.min())
                    _hi = float(_finite.max())
                    _edges = np.linspace(_lo, _hi, int(quantization_nbins) + 1)
                else:
                    _edges = np.linspace(0.0, 0.0, int(quantization_nbins) + 1)
            quantization["edges"] = _edges.tolist()

    # Per-operand pre-warp (2026-06-02): when the unary name on a side is the
    # learned ``prewarp`` pseudo-unary, persist its fitted spec (basis, degree,
    # coeffs, basis-preprocess params) FLAT in ``extra`` so replay reproduces
    # the closed-form 1-D warp from x alone (leak-safe; no y). ``extra`` values
    # must stay flat ndarray/scalar/str for ``_extra_equal`` -- the nested
    # ``preprocess`` dict is JSON-stringified (sorted keys, deterministic).
    extra: dict = {}
    for _side, _spec in (("a", prewarp_a), ("b", prewarp_b)):
        if _spec is None:
            continue
        extra[f"prewarp_{_side}_basis"] = str(_spec["basis"])
        extra[f"prewarp_{_side}_degree"] = int(_spec["degree"])
        extra[f"prewarp_{_side}_coef"] = np.asarray(_spec["coef"], dtype=np.float64).copy()
        extra[f"prewarp_{_side}_preprocess"] = _orjson_pp(_spec["preprocess"])
    # Per-operand median gate (2026-06-04): when a side used the ``gate_med``
    # pseudo-unary, persist its single TRAIN-median float FLAT in ``extra`` so
    # replay reproduces ``(x > median)`` from x alone (leak-safe; no y). A plain
    # float is JSON-friendly and ``_extra_equal``-comparable as a scalar.
    for _side, _med in (("a", gate_med_a), ("b", gate_med_b)):
        if _med is None:
            continue
        extra[f"gate_med_{_side}_median"] = float(_med)
    # Nested-engineered parents (2026-06-08): store the parent recipe per side so
    # replay recomputes the engineered operand recursively. ``_extra_equal`` compares
    # them via ``EngineeredRecipe.__eq__`` (the else branch), and ``__post_init__``
    # deep-copies extra so the stored parents are owned by this recipe.
    for _side, _parent in (("a", nested_parent_a), ("b", nested_parent_b)):
        if _parent is None:
            continue
        extra[f"nested_parent_{_side}"] = _parent
    return EngineeredRecipe(
        name=name,
        kind="unary_binary",
        src_names=(src_a_name, src_b_name),
        unary_names=(unary_a_name, unary_b_name),
        binary_name=binary_name,
        unary_preset=unary_preset,
        binary_preset=binary_preset,
        quantization=quantization,
        extra=extra,
    )


def _orjson_pp(d: dict) -> str:
    """Serialise a flat preprocess-params dict to a deterministic JSON string
    (sorted keys) for storage in ``EngineeredRecipe.extra``. orjson keeps the
    recipe pickle-light and the OPT_SORT_KEYS flag makes the stored bytes a
    stable function of the params (hash-friendly)."""
    import orjson
    return orjson.dumps(d, option=orjson.OPT_SORT_KEYS).decode("ascii")





# ---------------------------------------------------------------------------
# Layer 56 (2026-05-31): orth_triplet_cross recipe builder re-export.
# Implementation in sibling ``_orthogonal_triplet_fe_recipes`` keeps the
# parent module under the 1700-line budget without hiding the builder from
# callers that already import everything from ``engineered_recipes``.
# ---------------------------------------------------------------------------
from .._orthogonal_triplet_fe_recipes import (  # noqa: E402
    build_orth_triplet_cross_recipe,
    _apply_orth_triplet_cross,
)

# ---------------------------------------------------------------------------
# Layer 77 (2026-06-01): orth_quadruplet_cross recipe builder re-export.
# Implementation in sibling ``_orthogonal_quadruplet_fe_recipes`` keeps the
# parent module under the 1700-line budget.
# ---------------------------------------------------------------------------
from .._orthogonal_quadruplet_fe_recipes import (  # noqa: E402
    build_orth_quadruplet_cross_recipe,
    _apply_orth_quadruplet_cross,
)

# ---------------------------------------------------------------------------
# Layer 90 (2026-06-01): numeric-decomposition recipe builders re-export.
# Implementation in sibling ``_numeric_decompose_fe`` keeps this module from
# growing further; the apply path is dispatched inline above (pure arithmetic).
# ---------------------------------------------------------------------------
from .._numeric_decompose_fe import (  # noqa: E402
    build_numeric_rounding_recipe,
    build_digit_extract_recipe,
)

# ---------------------------------------------------------------------------
# Layer 92 (2026-06-01): temporal leak-safe aggregation recipe builders
# re-export. Implementation in sibling ``_temporal_agg_fe`` keeps this module
# from growing further; the apply path is dispatched inline above.
# ---------------------------------------------------------------------------
from .._temporal_agg_fe import (  # noqa: E402
    build_temporal_expanding_recipe,
    build_temporal_rolling_recipe,
    build_temporal_lag_recipe,
)

# ---------------------------------------------------------------------------
# Layer 95 PART A (2026-06-01): periodic / modular decomposition recipe builder
# re-export. Implementation in sibling ``_periodic_fe`` keeps this module from
# growing further; the apply path is dispatched inline above (pure arithmetic).
# ---------------------------------------------------------------------------
from .._periodic_fe import build_modular_recipe  # noqa: E402

# ---------------------------------------------------------------------------
# Layer 95 PART B (2026-06-01): per-group distribution-distance recipe builder
# re-export. Implementation in sibling ``_group_distance_fe`` keeps this module
# from growing further; the apply path is dispatched inline above.
# ---------------------------------------------------------------------------
from .._group_distance_fe import build_group_distance_recipe  # noqa: E402

# Orthogonal-basis recipe builders + replay helpers live in the sibling
# ``_orth_basis_recipes`` (apply path dispatched lazily above); re-exported so
# external importers keep resolving them from this module.
from ._orth_basis_recipes import (  # noqa: E402,F401
    _apply_orth_fourier,
    _apply_orth_pair_cross,
    _apply_orth_pre_transform,
    _apply_orth_spline,
    _apply_orth_univariate,
    _bspline_basis_values,
    _eval_orth_basis_column,
    _fit_spline_knots,
    build_orth_cluster_basis_recipe,
    build_orth_diff_basis_recipe,
    build_orth_fourier_recipe,
    build_orth_pair_cross_recipe,
    build_orth_spline_recipe,
    build_orth_univariate_recipe,
)

# Categorical-encoding recipe builders + replay helpers live in the sibling
# ``_encoding_recipes`` (apply path dispatched lazily above); re-exported so
# external importers keep resolving them from this module.
from ._encoding_recipes import (  # noqa: E402,F401
    _apply_cat_num_residual,
    _apply_count_encoded,
    _apply_frequency_encoded,
    _apply_kfold_target_encoded,
    build_cat_num_residual_recipe,
    build_cat_pair_cross_recipe,
    build_cat_triple_cross_recipe,
    build_count_encoded_recipe,
    build_frequency_encoded_recipe,
    build_kfold_target_encoded_recipe,
)

# MI-greedy + missingness / pairwise-ratio recipe builders live in the sibling
# ``_missingness_ratio_recipes`` (mi-greedy apply dispatched lazily above);
# re-exported so external importers keep resolving them from this module.
from ._missingness_ratio_recipes import (  # noqa: E402,F401
    _apply_mi_greedy_transform,
    build_missing_indicator_recipe,
    build_missingness_count_recipe,
    build_missingness_pattern_recipe,
    build_mi_greedy_transform_recipe,
    build_pairwise_ratio_recipe,
)

# Grouped / temporal recipe builders live in the sibling ``_grouped_recipes``
# (apply helpers dispatched lazily above); re-exported for external importers.
from ._grouped_recipes import (  # noqa: E402,F401
    build_composite_group_agg_recipe,
    build_grouped_agg_recipe,
    build_grouped_delta_recipe,
    build_grouped_quantile_recipe,
    build_lagged_diff_recipe,
    build_target_aware_group_bin_recipe,
)

