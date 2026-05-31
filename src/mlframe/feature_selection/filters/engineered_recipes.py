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
    kind: Literal["unary_binary", "factorize", "hermite_pair", "target_encoding", "cluster_aggregate", "orth_univariate", "orth_pair_cross", "orth_spline", "orth_fourier", "mi_greedy_transform", "kfold_target_encoded"]
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
    if recipe.kind == "orth_spline":
        return _apply_orth_spline(recipe, X)
    if recipe.kind == "orth_fourier":
        return _apply_orth_fourier(recipe, X)
    if recipe.kind == "mi_greedy_transform":
        return _apply_mi_greedy_transform(recipe, X)
    if recipe.kind == "kfold_target_encoded":
        return _apply_kfold_target_encoded(recipe, X)
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

    if method == "median":
        out = np.median(Z, axis=1)
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


def _eval_orth_basis_column(x: np.ndarray, basis: str, degree: int) -> np.ndarray:
    """Preprocess x to the basis domain (z-score for hermite, min-max for
    legendre/chebyshev, shift for laguerre), then evaluate the single basis
    function of the given degree via a one-hot coefficient vector.

    Mirrors ``_orthogonal_univariate_fe._evaluate_basis_column`` so that
    transform()-time replay produces the SAME value as fit-time generation.
    Lazy import of ``hermite_fe`` keeps the recipes module dependency-light.
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
    vals = _extract_column(X, src_name)
    return _eval_orth_basis_column(vals, basis, degree)


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
    *, name: str, src_name: str, basis: str, degree: int,
) -> EngineeredRecipe:
    """Frozen recipe for one orthogonal-polynomial univariate column
    ``basis_n(preprocess(X[src_name]))``. Replay is closed-form and
    deterministic; no y reference is captured."""
    return EngineeredRecipe(
        name=name,
        kind="orth_univariate",
        src_names=(src_name,),
        extra={"basis": str(basis), "degree": int(degree)},
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
