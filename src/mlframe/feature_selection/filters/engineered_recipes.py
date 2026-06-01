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
from typing import Any, Literal

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
    kind: Literal["unary_binary", "factorize", "hermite_pair", "target_encoding", "cluster_aggregate", "orth_univariate", "orth_pair_cross", "orth_triplet_cross", "orth_quadruplet_cross", "orth_spline", "orth_fourier", "orth_diff_basis", "orth_cluster_basis", "mi_greedy_transform", "kfold_target_encoded", "count_encoded", "frequency_encoded", "cat_num_residual", "missing_indicator", "missingness_count", "missingness_pattern", "pairwise_ratio", "grouped_delta", "lagged_diff", "grouped_agg", "grouped_quantile", "target_aware_group_bin", "cat_pair_cross", "numeric_rounding", "digit_extract"]
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
        return _apply_orth_univariate(recipe, X)
    if recipe.kind == "orth_pair_cross":
        return _apply_orth_pair_cross(recipe, X)
    if recipe.kind == "orth_diff_basis":
        # Layer 59 (2026-05-31): lazy import keeps this module under the
        # ~1.8k-LOC ceiling; the apply helper lives in the sibling FE module.
        from ._orthogonal_diff_basis_fe import _apply_orth_diff_basis
        return _apply_orth_diff_basis(recipe, X)
    if recipe.kind == "orth_cluster_basis":
        # Layer 61 (2026-05-31): per-cluster shared-basis FE. Replay
        # recomputes the aggregate from the stored member tuple via the
        # recipe-stored aggregator (mean_z / median_z / pc1), then evaluates
        # the same basis_degree -- bit-exact round-trip from fit to transform.
        from ._orthogonal_cluster_basis_fe import _apply_orth_cluster_basis
        return _apply_orth_cluster_basis(recipe, X)
    if recipe.kind == "orth_triplet_cross":
        from ._orthogonal_triplet_fe_recipes import _apply_orth_triplet_cross
        return _apply_orth_triplet_cross(recipe, X)
    if recipe.kind == "orth_quadruplet_cross":
        # Layer 77 (2026-06-01): 4-way cross-basis FE. Replay is closed-form
        # over the four source columns via the recipe-stored (basis, deg) tuple.
        from ._orthogonal_quadruplet_fe_recipes import _apply_orth_quadruplet_cross
        return _apply_orth_quadruplet_cross(recipe, X)
    if recipe.kind == "orth_spline":
        return _apply_orth_spline(recipe, X)
    if recipe.kind == "orth_fourier":
        return _apply_orth_fourier(recipe, X)
    if recipe.kind == "mi_greedy_transform":
        return _apply_mi_greedy_transform(recipe, X)
    if recipe.kind == "kfold_target_encoded":
        return _apply_kfold_target_encoded(recipe, X)
    if recipe.kind == "count_encoded":
        return _apply_count_encoded(recipe, X)
    if recipe.kind == "frequency_encoded":
        return _apply_frequency_encoded(recipe, X)
    if recipe.kind == "cat_num_residual":
        return _apply_cat_num_residual(recipe, X)
    if recipe.kind in ("missing_indicator", "missingness_count", "missingness_pattern"):
        # Layer 37 lazy import: keeps engineered_recipes.py under the 1k-LOC
        # ceiling (mlframe sibling-split rule). The helpers live alongside
        # the encoders that build them.
        from ._missingness_fe import (
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
        from ._ratio_delta_fe import (
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
        from ._grouped_agg_fe import _apply_grouped_agg_recipe
        return _apply_grouped_agg_recipe(recipe, X)
    if recipe.kind == "cat_pair_cross":
        # Layer 89 lazy import: cat x cat synergy cross; replay helper lives
        # with the generator. Keeps this module under the LOC ceiling.
        from ._cat_pair_fe import apply_cat_pair_cross
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
    if recipe.kind in ("numeric_rounding", "digit_extract"):
        # Layer 90 (2026-06-01): numeric decomposition (multi-precision
        # rounding + decimal-digit extraction). Pure arithmetic on the single
        # source column -- no lazy import needed, no y reference.
        src_name = recipe.src_names[0]
        vals = _extract_column(X, src_name)
        if recipe.kind == "numeric_rounding":
            from ._numeric_decompose_fe import apply_rounding
            return apply_rounding(vals, float(recipe.extra["precision"]))
        from ._numeric_decompose_fe import apply_digit_extract
        return apply_digit_extract(vals, int(recipe.extra["digit_position"]))
    if recipe.kind in ("grouped_quantile", "target_aware_group_bin"):
        # Layer 88 lazy import: per-group distributional FE (percentile-rank +
        # spread + target-aware supervised bins); replay helpers live with the
        # generator. Keeps this module under the LOC ceiling.
        from ._grouped_quantile_fe import (
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
    from .hermite_fe import _POLY_BASES, _DEFAULT_BIN_FUNCS

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
    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)

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
        from .discretization import discretize_array
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
    from .feature_engineering import create_unary_transformations, create_binary_transformations
    from .discretization import discretize_array

    unary_funcs = create_unary_transformations(preset=recipe.unary_preset)
    binary_funcs = create_binary_transformations(preset=recipe.binary_preset)

    name_a, name_b = recipe.src_names
    u_a, u_b = recipe.unary_names

    if u_a not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_a}' not in '{recipe.unary_preset}' preset. "
            f"Replay requires the same preset that was active at fit time."
        )
    if u_b not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_b}' not in '{recipe.unary_preset}' preset."
        )
    if recipe.binary_name not in binary_funcs:
        raise KeyError(
            f"Binary function '{recipe.binary_name}' not in "
            f"'{recipe.binary_preset}' preset."
        )

    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)

    transformed_a = unary_funcs[u_a](vals_a)
    transformed_b = unary_funcs[u_b](vals_b)
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
) -> EngineeredRecipe:
    """Build an ``EngineeredRecipe`` of kind ``"unary_binary"``. ``quantization`` is ``None`` if no discretization, else a dict carrying the binning
    parameters AND, when ``fit_values_for_edges`` is provided, the fit-time bin edges so transform-time replay maps each row to the SAME bin
    code regardless of test-data distribution. Dtype is stringified so the recipe is JSON-friendly and pickle-safe across numpy versions.

    2026-05-30 Wave 9.1 iter 28: ``fit_values_for_edges`` lets the caller pin the quantile boundaries. Without it (legacy code paths) the
    recipe emits a UserWarning at replay time about the train/test leakage risk.
    """
    if quantization_nbins is None:
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
    return EngineeredRecipe(
        name=name,
        kind="unary_binary",
        src_names=(src_a_name, src_b_name),
        unary_names=(unary_a_name, unary_b_name),
        binary_name=binary_name,
        unary_preset=unary_preset,
        binary_preset=binary_preset,
        quantization=quantization,
    )


# ---------------------------------------------------------------------------
# Layer 23 (2026-05-31): orthogonal-polynomial univariate / pair-cross recipes
# ---------------------------------------------------------------------------
#
# These replay the engineered columns produced by ``hybrid_orth_mi_fe`` /
# ``hybrid_orth_mi_pair_fe`` from ``_orthogonal_univariate_fe.py``. Both
# routes are CLOSED-FORM functions of the source column(s) alone -- the MI
# scoring that picked them at fit time consumed y, but the column value
# itself does not depend on y. Replay therefore reads only X.
#
# extra layout:
# * orth_univariate : {basis: str, degree: int}
# * orth_pair_cross : {basis_i: str, basis_j: str, deg_a: int, deg_b: int}


def _apply_orth_pre_transform(x: np.ndarray, pre_transform: str) -> np.ndarray:
    """Layer 58 (2026-05-31) per-column pre-transform: optionally reshape the
    raw column BEFORE the basis-domain preprocess kicks in. Supported values:

    * ``"raw"``      -- identity (no pre-transform; legacy Layer 21/57 path)
    * ``"log_abs"``  -- ``log(|x| + 1e-12)`` -- captures heavy-tail log-normal
                        targets where raw Hermite z-score collapses the signal.
    * ``"sqrt_abs"`` -- ``sign(x) * sqrt(|x|)`` -- mild non-linear stretch.
    * ``"tanh"``     -- ``tanh(x / max(std, 1e-12))`` -- bounded mapping; pairs
                        well with Chebyshev/Legendre on otherwise-unbounded
                        inputs.

    Replay invariant: stateless given x + pre_transform name. The std for
    ``tanh`` is computed on-the-fly from the SAME column so train/test parity
    holds (z-score in the basis preprocess already does the same). Unknown
    values fall back to identity with a logger warning rather than raising.
    """
    x = np.asarray(x, dtype=np.float64)
    if pre_transform in (None, "", "raw", "identity"):
        return x
    if pre_transform == "log_abs":
        return np.log(np.abs(x) + 1e-12)
    if pre_transform == "sqrt_abs":
        return np.sign(x) * np.sqrt(np.abs(x))
    if pre_transform == "tanh":
        sd = float(np.std(x))
        return np.tanh(x / sd) if sd > 1e-12 else np.tanh(x)
    # Unknown -- warn and fall back to identity. Avoids raising at replay
    # time on pickles produced by older clients that introduced bespoke
    # tags; the worst case is one orth column that's the wrong shape, which
    # downstream MRMR will deselect on the next refit.
    import logging as _lg_pretrans
    _lg_pretrans.getLogger(__name__).warning(
        "_apply_orth_pre_transform: unknown pre_transform %r; falling back "
        "to identity.", pre_transform,
    )
    return x


def _eval_orth_basis_column(
    x: np.ndarray,
    basis: str,
    degree: int,
    *,
    pre_transform: str = "raw",
) -> np.ndarray:
    """Preprocess x to the basis domain (z-score for hermite, min-max for
    legendre/chebyshev, shift for laguerre), then evaluate the single basis
    function of the given degree via a one-hot coefficient vector.

    Mirrors ``_orthogonal_univariate_fe._evaluate_basis_column`` so that
    transform()-time replay produces the SAME value as fit-time generation.
    Lazy import of ``hermite_fe`` keeps the recipes module dependency-light.

    Layer 58 (2026-05-31): optional ``pre_transform`` applied to the column
    BEFORE the basis preprocess (log|x| for heavy-tail, tanh for bounded
    mapping, etc.). ``pre_transform='raw'`` (default) keeps Layer 21/57
    byte-identical -- existing recipes deserialized without the field
    behave unchanged.
    """
    from .hermite_fe import _POLY_BASES, polyeval_dispatch
    basis_info = _POLY_BASES[basis]
    fit_fn = basis_info["fit"]
    # NaN-safe: mirror fit-time finite-mask behaviour. Fit-time uses the
    # COLUMN mean to fill NaN before z-score / min-max. Test-time must use
    # the SAME fill so the basis-evaluation parity holds row-by-row when the
    # train and test frames disagree on which rows are NaN. The hybrid FE
    # path uses ``np.nanmean`` on the column at fit time; we replicate.
    x = np.asarray(x, dtype=np.float64)
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        x = np.where(finite_mask, x, fill)
    # Layer 58: optional pre-transform before basis preprocess. Identity
    # when ``pre_transform='raw'`` (default) -- legacy bit-exact path.
    x = _apply_orth_pre_transform(x, pre_transform)
    # The pre-transform can produce non-finite values for pathological inputs
    # (e.g. log(0) when caller passed a value at the floor); guard with one
    # more NaN-safe fill so the basis preprocess (z-score / min-max) doesn't
    # propagate NaN through every downstream column.
    finite_mask2 = np.isfinite(x)
    if not finite_mask2.all():
        fill2 = float(np.nanmean(x[finite_mask2])) if finite_mask2.any() else 0.0
        x = np.where(finite_mask2, x, fill2)
    z, _params = fit_fn(x)
    z = np.ascontiguousarray(z, dtype=np.float64)
    coef = np.zeros(int(degree) + 1, dtype=np.float64)
    coef[int(degree)] = 1.0
    return polyeval_dispatch(basis, z, coef)


def _apply_orth_univariate(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay an orthogonal-polynomial univariate column: extract the
    source column, evaluate basis_n(z) where z is the per-basis preprocessed
    value. Stateless given the stored basis + degree; no y reference.
    """
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"orth_univariate recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("basis", "degree"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_univariate recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    src_name = recipe.src_names[0]
    basis = str(recipe.extra["basis"])
    degree = int(recipe.extra["degree"])
    # Layer 58 (2026-05-31): optional pre-transform applied to the raw column
    # before the basis preprocess. Default ``"raw"`` (identity) keeps recipes
    # produced by Layer 21/57 byte-identical -- existing pickles missing the
    # ``pre_transform`` key replay unchanged.
    pre_transform = str(recipe.extra.get("pre_transform", "raw"))
    vals = _extract_column(X, src_name)
    return _eval_orth_basis_column(
        vals, basis, degree, pre_transform=pre_transform,
    )


def _apply_orth_pair_cross(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a pair-cross-basis column: extract both source columns,
    evaluate basis_a^{deg_a}(z_i) * basis_b^{deg_b}(z_j). Stateless given
    the stored bases + degrees; no y reference.
    """
    if len(recipe.src_names) != 2:
        raise ValueError(
            f"orth_pair_cross recipe '{recipe.name}' must have exactly 2 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("basis_i", "basis_j", "deg_a", "deg_b"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_pair_cross recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    name_i, name_j = recipe.src_names
    basis_i = str(recipe.extra["basis_i"])
    basis_j = str(recipe.extra["basis_j"])
    deg_a = int(recipe.extra["deg_a"])
    deg_b = int(recipe.extra["deg_b"])
    vals_i = _extract_column(X, name_i)
    vals_j = _extract_column(X, name_j)
    h_a = _eval_orth_basis_column(vals_i, basis_i, deg_a)
    h_b = _eval_orth_basis_column(vals_j, basis_j, deg_b)
    return h_a * h_b


def build_orth_univariate_recipe(
    *,
    name: str,
    src_name: str,
    basis: str,
    degree: int,
    pre_transform: str = "raw",
) -> EngineeredRecipe:
    """Frozen recipe for one orthogonal-polynomial univariate column
    ``basis_n(preprocess(pre_transform(X[src_name])))``. Replay is closed-form
    and deterministic; no y reference is captured.

    ``pre_transform`` defaults to ``"raw"`` (identity) so Layer 21/57
    recipes remain byte-identical. Layer 58 routing FE picks one of
    ``"raw" | "log_abs" | "sqrt_abs" | "tanh"`` per surviving column.
    """
    extra = {"basis": str(basis), "degree": int(degree)}
    # Only write the field when the caller picked a non-default value so
    # legacy pickles continue to compare equal byte-for-byte (the recipe's
    # ``__eq__`` walks ``extra`` content).
    if pre_transform and pre_transform != "raw":
        extra["pre_transform"] = str(pre_transform)
    return EngineeredRecipe(
        name=name,
        kind="orth_univariate",
        src_names=(src_name,),
        extra=extra,
    )


def build_orth_pair_cross_recipe(
    *, name: str, src_a_name: str, src_b_name: str,
    basis_i: str, basis_j: str, deg_a: int, deg_b: int,
) -> EngineeredRecipe:
    """Frozen recipe for one cross-basis pair column
    ``basis_i^{deg_a}(preprocess(X[a])) * basis_j^{deg_b}(preprocess(X[b]))``.
    """
    return EngineeredRecipe(
        name=name,
        kind="orth_pair_cross",
        src_names=(src_a_name, src_b_name),
        extra={
            "basis_i": str(basis_i),
            "basis_j": str(basis_j),
            "deg_a": int(deg_a),
            "deg_b": int(deg_b),
        },
    )


def build_orth_diff_basis_recipe(
    *, name: str, col_a: str, col_b: str,
    basis: str, degree: int, pre_transform: str = "raw",
) -> EngineeredRecipe:
    """Layer 59 (2026-05-31): frozen recipe for one diff-basis column
    ``basis_degree(preprocess(pre_transform(X[col_a] - X[col_b])))``.

    The diff orientation is FIXED as ``col_a - col_b`` so the recipe replays
    deterministically; reversing the column order yields a sign-flipped
    column which the MI scorer treats as a distinct candidate. Replay is a
    pure function of X (no y reference). ``pre_transform`` defaults to
    ``"raw"`` so the field is omitted from ``extra`` on the legacy path,
    keeping recipe byte-equality with earlier diff-basis pickles that pre-
    date the pre-transform feature.
    """
    extra = {"basis": str(basis), "degree": int(degree)}
    if pre_transform and pre_transform != "raw":
        extra["pre_transform"] = str(pre_transform)
    return EngineeredRecipe(
        name=name,
        kind="orth_diff_basis",
        src_names=(str(col_a), str(col_b)),
        extra=extra,
    )


def build_orth_cluster_basis_recipe(
    *, name: str, members: tuple[str, ...],
    basis: str, degree: int, aggregator: str = "mean_z",
) -> EngineeredRecipe:
    """Layer 61 (2026-05-31): frozen recipe for one per-cluster shared-
    basis column ``basis_degree(preprocess(aggregator(members)))``.

    The member tuple is stored in deterministic (sorted) order so the
    aggregate orientation matches fit time exactly. ``aggregator`` is
    one of ``mean_z`` / ``median_z`` / ``pc1`` -- see
    :func:`compute_cluster_aggregate` in the cluster-basis FE module.
    Replay is a pure function of X (no y reference).
    """
    if len(members) < 2:
        raise ValueError(
            f"build_orth_cluster_basis_recipe: ``members`` must have >=2 "
            f"entries (cluster, not singleton); got {len(members)}."
        )
    return EngineeredRecipe(
        name=name,
        kind="orth_cluster_basis",
        src_names=tuple(str(m) for m in members),
        extra={
            "basis": str(basis),
            "degree": int(degree),
            "aggregator": str(aggregator),
        },
    )


# ---------------------------------------------------------------------------
# Layer 26 (2026-05-31): generic MI-greedy transform recipes
# ---------------------------------------------------------------------------
#
# Replays the engineered columns produced by ``greedy_mi_fe_construct`` in
# ``_mi_greedy_fe.py``. Same property as ``orth_univariate`` /
# ``orth_pair_cross``: the column value is a closed-form function of the
# source column(s) alone (the MI scorer that picked the candidate at fit time
# consumed y, but the column expression doesn't). Replay reads only X.
#
# extra layout:
# * mi_greedy_transform: {transform: str}
# The src_names tuple length (1 or 2) determines unary vs binary at replay
# time; the registry lookup in ``apply_mi_greedy_transform`` enforces the
# transform/arity contract.


def _apply_mi_greedy_transform(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a generic MI-greedy engineered column. Stateless given the
    stored transform name + source columns; no y reference."""
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
    from ._mi_greedy_fe import apply_mi_greedy_transform
    src_values = [
        np.asarray(_extract_column(X, n), dtype=np.float64)
        for n in recipe.src_names
    ]
    return apply_mi_greedy_transform(str(recipe.extra["transform"]), src_values)


def build_mi_greedy_transform_recipe(
    *, name: str, transform: str, src_names: tuple[str, ...],
) -> EngineeredRecipe:
    """Frozen recipe for one generic MI-greedy engineered column. ``transform``
    must be one of the keys in ``_mi_greedy_fe.UNARY_TRANSFORMS`` (or
    ``TRIG_BOUNDED_TRANSFORMS``) for unary recipes, or
    ``_mi_greedy_fe.BINARY_TRANSFORMS`` for binary recipes; the registry
    lookup at replay time enforces the validation."""
    return EngineeredRecipe(
        name=name,
        kind="mi_greedy_transform",
        src_names=tuple(src_names),
        extra={"transform": str(transform)},
    )


# ---------------------------------------------------------------------------
# Layer 32 (2026-05-31): B-spline + Fourier basis FE recipes
# ---------------------------------------------------------------------------
#
# Spline / Fourier complement the orthogonal-polynomial bases (Hermite /
# Legendre / Chebyshev / Laguerre): they cover signal shapes the polynomial
# bases miss -- sharp local thresholds (B-spline) and periodic patterns
# (Fourier). Both replays are CLOSED-FORM functions of the source column
# alone -- the MI scoring that picked them at fit time consumed y, but the
# column value itself does not depend on y. Replay therefore reads only X.
#
# extra layout:
# * orth_spline  : {knots: ndarray[float64], idx: int, lo: float, hi: float}
#                  Cubic B-spline basis B_idx(z) where z = (x - lo) / (hi - lo)
#                  clipped to [0, 1]; knots are quantile-placed at fit time.
# * orth_fourier : {kind: "sin"|"cos", freq: float, lo: float, span: float}
#                  sin(2*pi*freq*z) or cos(2*pi*freq*z) where z = (x - lo) / span.


def _bspline_basis_values(z: np.ndarray, knots: np.ndarray, idx: int, degree: int = 3) -> np.ndarray:
    """Evaluate the ``idx``-th cubic B-spline basis function at points ``z``.

    Uses the Cox-de Boor recursion. ``knots`` is the full augmented knot
    vector (with degree+1 repeated boundary knots). Returns shape (n,).
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    # Degree-0 indicator [t_k, t_{k+1})
    # Build up the Cox-de Boor recursion for the single basis index `idx`.
    # We do it in O(degree+1) per point by computing the full row of B-splines
    # via the standard algorithm, then picking the column we need.
    nk = len(knots)
    for i in range(n):
        zi = z[i]
        # Find span: largest k with knots[k] <= zi < knots[k+1]
        # Handle boundary: clip zi to [knots[degree], knots[-degree-1]]
        if zi >= knots[nk - degree - 1]:
            zi_eff = knots[nk - degree - 1] - 1e-12
        elif zi <= knots[degree]:
            zi_eff = knots[degree] + 1e-12
        else:
            zi_eff = zi
        # Compute non-zero B-splines of given degree at zi_eff using standard
        # de Boor algorithm (returns degree+1 non-zero values starting at span k-degree)
        # Locate k such that knots[k] <= zi_eff < knots[k+1]
        k = degree
        for kk in range(degree, nk - degree - 1):
            if knots[kk] <= zi_eff < knots[kk + 1]:
                k = kk
                break
        else:
            k = nk - degree - 2
        # B-splines of degree d at zi_eff for span k.
        # left[j] = zi - knots[k+1-j]; right[j] = knots[k+j] - zi
        # N[0] = 1, then for d = 1..degree update.
        N = np.zeros(degree + 1, dtype=np.float64)
        N[0] = 1.0
        for d in range(1, degree + 1):
            saved = 0.0
            for r in range(d):
                t_left = knots[k + 1 + r - d]
                t_right = knots[k + 1 + r]
                denom = t_right - t_left
                if denom <= 1e-12:
                    temp = 0.0
                else:
                    temp = N[r] / denom
                N[r] = saved + (t_right - zi_eff) * temp
                saved = (zi_eff - t_left) * temp
            N[d] = saved
        # Non-zero basis indices for this span are k-degree..k.
        # The column we want is `idx`. If idx in [k-degree, k] return N[idx - (k-degree)], else 0.
        rel = idx - (k - degree)
        if 0 <= rel <= degree:
            out[i] = N[rel]
        # else: 0 (already initialized)
    return out


def _fit_spline_knots(x: np.ndarray, n_inner_knots: int, degree: int = 3) -> tuple[np.ndarray, float, float]:
    """Fit a cubic B-spline knot vector at quantiles ``(1..K)/(K+1)`` of x.

    Returns the FULL augmented knot vector (with degree+1 boundary
    repetitions at 0 and 1) plus the (lo, hi) normalisation range.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        lo, hi = 0.0, 1.0
        inner = np.linspace(1.0 / (n_inner_knots + 1), n_inner_knots / (n_inner_knots + 1), n_inner_knots)
    else:
        xf = x[finite]
        lo, hi = float(xf.min()), float(xf.max())
        if hi - lo <= 1e-12:
            hi = lo + 1.0
        z = (xf - lo) / (hi - lo)
        qs = np.linspace(1.0 / (n_inner_knots + 1), n_inner_knots / (n_inner_knots + 1), n_inner_knots)
        inner = np.quantile(z, qs)
        # Ensure strictly increasing knots
        inner = np.unique(inner)
        if inner.size < n_inner_knots:
            # Top up with uniform fill where quantiles collapsed (ties).
            extra_n = n_inner_knots - inner.size
            uni = np.linspace(0.0, 1.0, extra_n + 2)[1:-1]
            inner = np.unique(np.concatenate([inner, uni]))
        inner = np.clip(inner, 1e-6, 1.0 - 1e-6)
    # Augment with degree+1 boundary repetitions on each end (clamped knot vector).
    boundary_lo = np.zeros(degree + 1, dtype=np.float64)
    boundary_hi = np.ones(degree + 1, dtype=np.float64)
    knots = np.concatenate([boundary_lo, inner.astype(np.float64), boundary_hi])
    return knots, float(lo), float(hi)


def _apply_orth_spline(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay one cubic B-spline basis column: B_{idx}(z) where z is min-max
    normalised x with knots fixed at fit time. Stateless given the stored
    knots + idx + (lo, hi); no y reference.
    """
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"orth_spline recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("knots", "idx", "lo", "hi"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_spline recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    name = recipe.src_names[0]
    knots = np.asarray(recipe.extra["knots"], dtype=np.float64)
    idx = int(recipe.extra["idx"])
    lo = float(recipe.extra["lo"])
    hi = float(recipe.extra["hi"])
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    span = max(hi - lo, 1e-12)
    z = np.clip((vals - lo) / span, 0.0, 1.0)
    return _bspline_basis_values(z, knots, idx, degree=3)


def _apply_orth_fourier(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay one Fourier basis column: sin(2*pi*freq*z) or cos(2*pi*freq*z)
    where z = (x - lo) / span, with (lo, span) fixed at fit time.
    """
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"orth_fourier recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("kind", "freq", "lo", "span"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_fourier recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    name = recipe.src_names[0]
    kind = str(recipe.extra["kind"])
    freq = float(recipe.extra["freq"])
    lo = float(recipe.extra["lo"])
    span = float(recipe.extra["span"])
    span = max(span, 1e-12)
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    z = (vals - lo) / span
    ang = 2.0 * np.pi * freq * z
    if kind == "sin":
        return np.sin(ang)
    if kind == "cos":
        return np.cos(ang)
    raise ValueError(f"orth_fourier recipe '{recipe.name}': unknown kind {kind!r}")


def build_orth_spline_recipe(
    *, name: str, src_name: str, knots: np.ndarray, idx: int, lo: float, hi: float,
) -> EngineeredRecipe:
    """Frozen recipe for one cubic B-spline basis column ``B_{idx}(z)`` where
    ``z = clip((X[src_name] - lo) / (hi - lo), 0, 1)`` with quantile-placed
    knots fixed at fit time."""
    return EngineeredRecipe(
        name=name,
        kind="orth_spline",
        src_names=(src_name,),
        extra={
            "knots": np.asarray(knots, dtype=np.float64).copy(),
            "idx": int(idx),
            "lo": float(lo),
            "hi": float(hi),
        },
    )


def build_orth_fourier_recipe(
    *, name: str, src_name: str, kind: str, freq: float, lo: float, span: float,
) -> EngineeredRecipe:
    """Frozen recipe for one Fourier basis column ``sin(2*pi*freq*z)`` or
    ``cos(2*pi*freq*z)`` where ``z = (X[src_name] - lo) / span`` with
    (lo, span) fixed at fit time."""
    if kind not in ("sin", "cos"):
        raise ValueError(f"orth_fourier kind must be 'sin' or 'cos'; got {kind!r}")
    return EngineeredRecipe(
        name=name,
        kind="orth_fourier",
        src_names=(src_name,),
        extra={
            "kind": str(kind),
            "freq": float(freq),
            "lo": float(lo),
            "span": float(span),
        },
    )


# ---------------------------------------------------------------------------
# Layer 33 (2026-05-31): K-fold target encoding for raw categorical columns
# ---------------------------------------------------------------------------
#
# The K-fold OOF mean-of-y per category. The recipe stores the FULL-DATA
# per-category mean (computed once at fit time) so transform() is a pure
# dict lookup with no y reference -- the OOF discipline is enforced at FIT
# only (the rows that fed into MRMR screening saw an out-of-fold encoded
# value, never their own y).
#
# extra layout:
# * kfold_target_encoded: {lookup: dict[str, float], global_mean: float,
#                          smoothing: float}
#   ``lookup`` maps the string-coerced source category to its smoothed
#   per-category mean. Categories not in the lookup at transform time map
#   to ``global_mean`` (no NaN propagation).
#
# Sibling to the cell-level ``target_encoding`` recipe above (which encodes
# MERGED k-way categorical CELLS inside the cat-FE pair-search kernel);
# this one encodes raw single columns and is the standard prod-ML pattern.


def _apply_kfold_target_encoded(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay a k-fold target-encoded column. Stateless given the stored
    lookup + global_mean; no y reference. Categories not present in the
    lookup map to ``global_mean``."""
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"kfold_target_encoded recipe '{recipe.name}' must have exactly "
            f"1 src_names; got {len(recipe.src_names)}"
        )
    for key in ("lookup", "global_mean"):
        if key not in recipe.extra:
            raise KeyError(
                f"kfold_target_encoded recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    # Lazy import to avoid circular dependency at module-load time.
    from ._target_encoding_fe import apply_target_encoding

    name = recipe.src_names[0]
    # Build a one-column frame view so apply_target_encoding's
    # column-name interface works on any X (pandas / polars / structured).
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and isinstance(X, pl.DataFrame):
        X_view = pd.DataFrame({name: X[name].to_numpy()})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: X[name]})
    else:
        raise TypeError(
            f"kfold_target_encoded recipe '{recipe.name}': cannot extract "
            f"column {name!r} from X of type {type(X).__name__}. Pass a "
            f"pandas / polars frame or a structured ndarray."
        )
    # Hand off to the apply helper. recipe.extra is a MappingProxy; pass a
    # plain dict so older helpers that index into it without abstract-base-
    # class checks don't trip.
    return apply_target_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "global_mean": float(recipe.extra["global_mean"]),
        },
    )


def build_kfold_target_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, global_mean: float,
    smoothing: float,
) -> EngineeredRecipe:
    """Frozen recipe for a single-column K-fold target-encoded column.

    ``lookup`` is the per-category mean-of-y table built from the full
    training data with Micci-Barreca smoothing. ``global_mean`` is the
    unconditional mean of y (used as the prior in smoothing AND as the
    fallback for unseen categories at transform). ``smoothing`` is stored
    for diagnostics / round-trip but is NOT consulted at transform time
    (smoothing already baked into ``lookup`` values).
    """
    # Coerce keys to plain str so MappingProxy round-trip + pickle work
    # cleanly even when the caller passed numpy/pandas string types.
    lookup_clean = {str(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="kfold_target_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "global_mean": float(global_mean),
            "smoothing": float(smoothing),
        },
    )


# ---------------------------------------------------------------------------
# Layer 34 (2026-05-31): count / frequency encoding + cat x num residual.
# ---------------------------------------------------------------------------
#
# Companions to ``kfold_target_encoded``: these three kinds cover the
# remaining production-grade categorical pipelines that Layer 33 does not
# touch. The kernels live in ``_count_freq_interaction_fe.py``.
#
# extra layout:
# * count_encoded     : {lookup: dict[str, int],   default: int}
# * frequency_encoded : {lookup: dict[str, float], default: float}
# * cat_num_residual  : {lookup: dict[str, float], global_mean: float,
#                        smoothing: float, num_col: str}
#   src_names is (cat_col, num_col) -- both extracted at replay.
#
# All three replays are STATELESS given X (no y reference, no fold info).


def _apply_count_encoded(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay count encoding via the stored per-category lookup."""
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"count_encoded recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    if "lookup" not in recipe.extra:
        raise KeyError(
            f"count_encoded recipe '{recipe.name}' missing 'lookup' in extra."
        )
    from ._count_freq_interaction_fe import apply_count_encoding

    name = recipe.src_names[0]
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and isinstance(X, pl.DataFrame):
        X_view = pd.DataFrame({name: X[name].to_numpy()})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: X[name]})
    else:
        raise TypeError(
            f"count_encoded recipe '{recipe.name}': cannot extract column "
            f"{name!r} from X of type {type(X).__name__}."
        )
    return apply_count_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "default": int(recipe.extra.get("default", 0)),
        },
    )


def _apply_frequency_encoded(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay frequency encoding via the stored per-category lookup."""
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"frequency_encoded recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    if "lookup" not in recipe.extra:
        raise KeyError(
            f"frequency_encoded recipe '{recipe.name}' missing 'lookup' in extra."
        )
    from ._count_freq_interaction_fe import apply_frequency_encoding

    name = recipe.src_names[0]
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and isinstance(X, pl.DataFrame):
        X_view = pd.DataFrame({name: X[name].to_numpy()})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: X[name]})
    else:
        raise TypeError(
            f"frequency_encoded recipe '{recipe.name}': cannot extract column "
            f"{name!r} from X of type {type(X).__name__}."
        )
    return apply_frequency_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "default": float(recipe.extra.get("default", 0.0)),
        },
    )


def _apply_cat_num_residual(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay cat x num residual: ``X[num] - lookup.get(X[cat], global_mean)``."""
    if len(recipe.src_names) != 2:
        raise ValueError(
            f"cat_num_residual recipe '{recipe.name}' must have exactly 2 "
            f"src_names (cat_col, num_col); got {len(recipe.src_names)}"
        )
    for key in ("lookup", "global_mean"):
        if key not in recipe.extra:
            raise KeyError(
                f"cat_num_residual recipe '{recipe.name}' missing {key!r} in extra."
            )
    from ._count_freq_interaction_fe import apply_cat_num_residual

    cat_name, num_name = recipe.src_names
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and isinstance(X, pl.DataFrame):
        X_view = pd.DataFrame({
            cat_name: X[cat_name].to_numpy(),
            num_name: X[num_name].to_numpy(),
        })
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({
            cat_name: X[cat_name],
            num_name: X[num_name],
        })
    else:
        raise TypeError(
            f"cat_num_residual recipe '{recipe.name}': cannot extract columns "
            f"from X of type {type(X).__name__}."
        )
    return apply_cat_num_residual(
        X_view, cat_name, num_name, {
            "lookup": dict(recipe.extra["lookup"]),
            "global_mean": float(recipe.extra["global_mean"]),
        },
    )


def build_count_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, default: int = 0,
) -> EngineeredRecipe:
    """Frozen recipe for a count-encoded column. ``lookup`` maps each
    fit-time category (as str) to its observed count; unseen categories at
    replay map to ``default`` (default 0). No y reference at any stage."""
    lookup_clean = {str(k): int(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="count_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "default": int(default),
        },
    )


def build_frequency_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, default: float = 0.0,
) -> EngineeredRecipe:
    """Frozen recipe for a frequency-encoded column (count / n_samples).
    Unseen categories at replay map to ``default`` (default 0.0)."""
    lookup_clean = {str(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="frequency_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "default": float(default),
        },
    )


def build_cat_num_residual_recipe(
    *, name: str, cat_name: str, num_name: str, lookup: dict,
    global_mean: float, smoothing: float,
) -> EngineeredRecipe:
    """Frozen recipe for a cat x num residual column. ``lookup`` is the
    smoothed per-category mean of ``num_name``; replay subtracts
    ``lookup.get(cat, global_mean)`` from the row's num value. Unseen
    categories fall back to ``global_mean`` (subtracting the unconditional
    mean is the natural fallback)."""
    lookup_clean = {str(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="cat_num_residual",
        src_names=(cat_name, num_name),
        extra={
            "lookup": lookup_clean,
            "global_mean": float(global_mean),
            "smoothing": float(smoothing),
            "num_col": str(num_name),
        },
    )


# ---------------------------------------------------------------------------
# Layer 89 (2026-06-01): cat x cat synergy cross. The (cat_i, cat_j) value-pair
# -> code mapping is stored as a list-of-(key, value) pairs so the frozen
# recipe's array-aware __eq__ / pickle round-trip handle the tuple keys cleanly
# (tuple-keyed dicts aren't JSON-friendly; a flat pair list is). ``encoding`` is
# "raw" (emit the integer cell code) or "target" (emit per-cell OOF mean-of-y
# from ``te_lookup``). The apply helper lives in ``_cat_pair_fe.py``.
# ---------------------------------------------------------------------------


def build_cat_pair_cross_recipe(
    *, name: str, cat_i: str, cat_j: str, mapping: dict,
    encoding: str = "raw", te_lookup: dict | None = None,
    global_mean: float = 0.0,
) -> EngineeredRecipe:
    """Frozen recipe for one cat x cat synergy cross.

    ``mapping`` maps each fit-time (str_i, str_j) value pair to its dense int
    cell code. ``encoding='raw'`` emits the cell code (unseen pairs -> sentinel
    bin); ``encoding='target'`` emits the per-cell smoothed mean-of-y from
    ``te_lookup`` (unseen pairs / codes -> ``global_mean``). No y reference is
    consumed at replay -- ``transform()`` is a pure function of X."""
    mapping_pairs = [
        [list(k), int(v)] for k, v in mapping.items()
    ]
    extra: dict = {
        "mapping": mapping_pairs,
        "encoding": str(encoding),
    }
    if encoding == "target":
        extra["te_lookup"] = [
            [int(k), float(v)] for k, v in (te_lookup or {}).items()
        ]
        extra["global_mean"] = float(global_mean)
    return EngineeredRecipe(
        name=name,
        kind="cat_pair_cross",
        src_names=(str(cat_i), str(cat_j)),
        extra=extra,
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


def build_grouped_delta_recipe(
    *, name: str, group_col: str, num_col: str, op: str,
    lookup_mean: dict, lookup_std: dict,
    global_mean: float, global_std: float,
) -> EngineeredRecipe:
    """Frozen recipe for a grouped-delta column. ``op='minus_mean'`` emits
    ``x - mean(x | group)``; ``op='div_std'`` emits the per-group z-score
    ``(x - mean(x | group)) / std(x | group)``. Both fall back to the
    train global mean / std when a group is unseen at replay."""
    if op not in ("minus_mean", "div_std"):
        raise ValueError(
            f"grouped_delta op must be 'minus_mean' or 'div_std'; got {op!r}"
        )
    lookup_mean_clean = {str(k): float(v) for k, v in lookup_mean.items()}
    lookup_std_clean = {str(k): float(v) for k, v in lookup_std.items()}
    return EngineeredRecipe(
        name=name,
        kind="grouped_delta",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "op": str(op),
            "lookup_mean": lookup_mean_clean,
            "lookup_std": lookup_std_clean,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


def build_grouped_agg_recipe(
    *, name: str, group_col: str, num_col: str, stat: str, op: str,
    group_lookup_dict: dict, global_value: float,
    lookup_mean: dict, lookup_std: dict,
    global_mean: float, global_std: float,
) -> EngineeredRecipe:
    """Layer 87 (2026-06-01): frozen recipe for one grouped multi-stat
    aggregate. ``op='broadcast'`` emits the per-group ``stat`` broadcast back
    to rows; ``op='z_within'`` emits ``(x - mean(x|group)) / std(x|group)``;
    ``op='ratio'`` emits ``x / mean(x|group)``. Unseen groups at replay fall
    back to the fit-time global statistic. Replay reads only X (no y), so
    transform() is leakage-free."""
    if op not in ("broadcast", "z_within", "ratio"):
        raise ValueError(
            f"grouped_agg op must be 'broadcast', 'z_within', or 'ratio'; "
            f"got {op!r}"
        )
    lookup_clean = {str(k): float(v) for k, v in group_lookup_dict.items()}
    lookup_mean_clean = {str(k): float(v) for k, v in lookup_mean.items()}
    lookup_std_clean = {str(k): float(v) for k, v in lookup_std.items()}
    return EngineeredRecipe(
        name=name,
        kind="grouped_agg",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "stat": str(stat),
            "op": str(op),
            "group_lookup_dict": lookup_clean,
            "global_value": float(global_value),
            "lookup_mean": lookup_mean_clean,
            "lookup_std": lookup_std_clean,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


def build_grouped_quantile_recipe(
    *, name: str, group_col: str, num_col: str, op: str,
    group_sorted: dict, global_sorted: list,
    iqr_lookup: dict, p90p10_lookup: dict,
    global_iqr: float, global_p90p10: float,
    quantiles=(),
) -> EngineeredRecipe:
    """Layer 88 (2026-06-01): frozen recipe for one per-group distributional
    feature. ``op='pct_rank'`` emits the empirical-CDF position of x within its
    group (stored per-group sorted value arrays); ``op='iqr'`` / ``op='p90p10'``
    emit the per-group spread broadcast. Unseen groups at replay fall back to
    the pooled global edges. Replay reads only X (no y), so transform() is
    leakage-free."""
    if op not in ("pct_rank", "iqr", "p90p10"):
        raise ValueError(
            f"grouped_quantile op must be 'pct_rank', 'iqr', or 'p90p10'; "
            f"got {op!r}"
        )
    group_sorted_clean = {
        str(k): [float(v) for v in vals] for k, vals in group_sorted.items()
    }
    return EngineeredRecipe(
        name=name,
        kind="grouped_quantile",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "op": str(op),
            "group_sorted": group_sorted_clean,
            "global_sorted": [float(v) for v in global_sorted],
            "iqr_lookup": {str(k): float(v) for k, v in iqr_lookup.items()},
            "p90p10_lookup": {str(k): float(v) for k, v in p90p10_lookup.items()},
            "global_iqr": float(global_iqr),
            "global_p90p10": float(global_p90p10),
            "quantiles": [float(q) for q in quantiles],
        },
    )


def build_target_aware_group_bin_recipe(
    *, name: str, group_col: str, num_col: str,
    group_edges: dict, global_edges: list, n_bins: int,
    op: str = "target_aware_bin",
) -> EngineeredRecipe:
    """Layer 88 (2026-06-01): frozen recipe for one target-aware per-group
    supervised bin index. ``group_edges`` holds, per group key, the inner MDLP
    edges (refit on ALL train rows, maximising ``I(bin; y)`` within the group);
    ``global_edges`` is the pooled fallback for unseen groups. Replay maps a
    row's value through ``searchsorted`` on its group's edges -- a pure function
    of X. The leak-safe OOF assignment used at fit for MI scoring is NOT
    persisted, so transform() carries no y reference."""
    group_edges_clean = {
        str(k): [float(v) for v in edges] for k, edges in group_edges.items()
    }
    return EngineeredRecipe(
        name=name,
        kind="target_aware_group_bin",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "group_edges": group_edges_clean,
            "global_edges": [float(v) for v in global_edges],
            "n_bins": int(n_bins),
        },
    )


def build_lagged_diff_recipe(
    *, name: str, time_col: str, value_col: str, period: int,
) -> EngineeredRecipe:
    """Frozen recipe for ``x_t - x_{t-period}`` after sorting by ``time_col``.
    Replay re-sorts the test frame by ``time_col`` and emits the per-row
    difference; the first ``period`` rows of the sorted order get 0."""
    if int(period) < 1:
        raise ValueError(f"lagged_diff period must be >= 1; got {period}")
    return EngineeredRecipe(
        name=name,
        kind="lagged_diff",
        src_names=(time_col, value_col),
        extra={
            "time_col": str(time_col),
            "value_col": str(value_col),
            "period": int(period),
        },
    )


# ---------------------------------------------------------------------------
# Layer 56 (2026-05-31): orth_triplet_cross recipe builder re-export.
# Implementation in sibling ``_orthogonal_triplet_fe_recipes`` keeps the
# parent module under the 1700-line budget without hiding the builder from
# callers that already import everything from ``engineered_recipes``.
# ---------------------------------------------------------------------------
from ._orthogonal_triplet_fe_recipes import (  # noqa: E402
    build_orth_triplet_cross_recipe,
    _apply_orth_triplet_cross,
)

# ---------------------------------------------------------------------------
# Layer 77 (2026-06-01): orth_quadruplet_cross recipe builder re-export.
# Implementation in sibling ``_orthogonal_quadruplet_fe_recipes`` keeps the
# parent module under the 1700-line budget.
# ---------------------------------------------------------------------------
from ._orthogonal_quadruplet_fe_recipes import (  # noqa: E402
    build_orth_quadruplet_cross_recipe,
    _apply_orth_quadruplet_cross,
)

# ---------------------------------------------------------------------------
# Layer 90 (2026-06-01): numeric-decomposition recipe builders re-export.
# Implementation in sibling ``_numeric_decompose_fe`` keeps this module from
# growing further; the apply path is dispatched inline above (pure arithmetic).
# ---------------------------------------------------------------------------
from ._numeric_decompose_fe import (  # noqa: E402
    build_numeric_rounding_recipe,
    build_digit_extract_recipe,
)

