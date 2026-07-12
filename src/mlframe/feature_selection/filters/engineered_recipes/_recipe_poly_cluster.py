"""Hermite-pair, cluster-aggregate, and target-encoding recipe builders + replay.

These three kinds share a "stored fit-time state, closed-form replay over X"
shape. ``_apply_hermite_pair`` lazy-imports ``hermite_fe`` and
``_apply_cluster_aggregate`` lazy-imports ``discretization`` inside the function
bodies to avoid an import cycle with those heavier siblings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _coerce_to_int_with_nan_handling, _extract_column


def _apply_hermite_pair(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """T1#3 2026-05-18: replay a Hermite/Chebyshev/Laguerre polynomial-pair FE column at predict time.

    Carries the full ``HermiteResult`` state (coefficients, basis name,
    bin-function name, preprocessing parameters) in ``recipe.extra``;
    the builder ``build_hermite_pair_recipe`` populates it. Lazy import
    of ``hermite_fe`` keeps this module dependency-light at import time.

    ``col_cache``: optional dict shared across every recipe replayed in ONE ``transform()`` call (see
    ``apply_recipe``); forwarded to ``_extract_column`` so a hub source column is pulled from ``X`` at most
    once per call.
    """
    if len(recipe.src_names) != 2:
        raise ValueError(f"hermite_pair recipe '{recipe.name}' must have exactly 2 src_names; " f"got {len(recipe.src_names)}")
    for key in ("coef_a", "coef_b", "basis", "bin_func_name", "preprocess_a", "preprocess_b"):
        if key not in recipe.extra:
            raise KeyError(
                f"hermite_pair recipe '{recipe.name}' missing '{key}' in extra. " f"Re-fit with the current build_hermite_pair_recipe to repopulate."
            )

    # Lazy imports to avoid circular dependency (hermite_fe -> mrmr -> recipes).
    from ..hermite_fe import _POLY_BASES, _DEFAULT_BIN_FUNCS

    basis = recipe.extra["basis"]
    bin_func_name = recipe.extra["bin_func_name"]
    if basis not in _POLY_BASES:
        raise KeyError(f"hermite_pair recipe '{recipe.name}' references unknown basis " f"{basis!r}; available: {sorted(_POLY_BASES)}")
    if bin_func_name not in _DEFAULT_BIN_FUNCS:
        raise KeyError(
            f"hermite_pair recipe '{recipe.name}' references unknown " f"bin_func_name {bin_func_name!r}; available: " f"{sorted(_DEFAULT_BIN_FUNCS)}"
        )

    name_a, name_b = recipe.src_names
    try:
        vals_a = _extract_column(X, name_a, col_cache=col_cache)
        vals_b = _extract_column(X, name_b, col_cache=col_cache)
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
    eval_dispatch: Any = basis_info["eval_dispatch"]
    coef_a = np.ascontiguousarray(recipe.extra["coef_a"], dtype=np.float64)
    coef_b = np.ascontiguousarray(recipe.extra["coef_b"], dtype=np.float64)
    # Unbounded polynomial-basis tails (Hermite/Chebyshev/Laguerre) can overflow on a replay row the
    # fit-time optimizer never saw (it only screens candidates for np.isfinite on the FIT sample -- see
    # hermite_fe's optimiser); ``bin_func`` (mul/add/sub) can then combine an overflowed leg into a
    # silent NaN with zero downstream scrub, unlike the sibling ``_apply_unary_binary`` replay.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h_a = eval_dispatch(z_a, coef_a)
        h_b = eval_dispatch(z_b, coef_b)
        bin_func: Any = _DEFAULT_BIN_FUNCS[bin_func_name]
        out = bin_func(h_a, h_b)
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(out, dtype=np.float64).reshape(-1)


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
    extra.update({k: v for k, v in (diagnostics or {}).items()})  # e.g. pca_var_ratio, representative -- scalars/str only
    # Build-time guard: _extra_equal compares values shallowly (np.array_equal / !=), so nested
    # lists/dicts of arrays would break __eq__/pickle round-trip. Keep extra flat.
    for k, v in extra.items():
        assert isinstance(v, (np.ndarray, str, int, float, bool)), f"cluster_aggregate extra[{k!r}] must be flat ndarray/scalar/str, got {type(v)}"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/engineered_recipes, not reachable with untrusted input
    return EngineeredRecipe(name=name, kind="cluster_aggregate", src_names=tuple(src_names), quantization=quantization, extra=extra)


def _apply_cluster_aggregate(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay a cluster-aggregate column: standardize members with stored train stats, sign-align,
    combine (``Z @ weights`` for linear methods, per-row median for ``median``), then discretize.

    Stateless given the stored ``extra`` -> uses ONLY train-fitted stats (never re-standardizes on the
    test distribution), so train/test parity holds. Pure numpy -> no lazy import / import-cycle risk.

    ``col_cache``: optional dict shared across every recipe replayed in ONE ``transform()`` call (see
    ``apply_recipe``); forwarded to ``_extract_column`` for every member column."""
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
            np.asarray(_extract_column(X, n, col_cache=col_cache), dtype=np.float64),
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
        out = np.sum(np.sign(Z) * (Z**2), axis=1)
    else:
        if "weights" not in recipe.extra:
            raise KeyError(f"cluster_aggregate recipe '{recipe.name}' (method={method!r}) missing 'weights' in extra.")
        out = Z @ np.asarray(recipe.extra["weights"], dtype=np.float64)

    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # CONTINUOUS TRANSFORM OUTPUT (2026-06-12): like its ``unary_binary`` sibling,
    # ``transform()`` delivers the cluster-aggregate column CONTINUOUS rather than as
    # the internal MI quantile code. Binning the continuous aggregate to integer codes
    # keeps only RANK and discards the MAGNITUDE that non-tree downstream models need:
    # measured on a target linear in a mean-z cluster aggregate, a linear model scored
    # test-R2 0.936 on the 10-bin code (Pearson 0.967 with the true aggregate) vs
    # 0.99972 on the continuous value (Pearson 1.000). This mirrors the
    # ``_recipe_unary_binary._apply_unary_binary`` change and the ``prewarp`` /
    # ``hermite_pair`` siblings that already skip replay-time quantization.
    # Continuous replay also SUBSUMES the Wave-9.1 iter-29 quantile-edge leak fix: the
    # output is a closed-form function of the operand row given the FROZEN
    # standardization (member_mean/member_std/signs/weights), so an identical physical
    # row maps to an identical value regardless of test-distribution drift -- there is
    # no quantile recomputation left to drift. ``recipe.quantization`` is kept for
    # provenance only; the downstream MRMR fit discretises the fit-time column for its
    # own MI matrix via ``_mrmr_fe_step`` (a separate path, unaffected).
    return np.asarray(out)


def _apply_target_encoding(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Look up each test row's (a, b) merged cell in ``cell_means``, return per-row encoded float. Unseen combinations map to ``global_mean`` (or per
    ``unknown_strategy``). ``col_cache``: optional dict shared across every recipe replayed in ONE ``transform()`` call (see ``apply_recipe``);
    forwarded to ``_extract_column``."""
    if len(recipe.src_names) != 2:
        raise NotImplementedError(f"target_encoding for k>2 not implemented yet " f"(recipe '{recipe.name}' has {len(recipe.src_names)} src).")
    if "cell_means" not in recipe.extra or "factorize_lookup" not in recipe.extra:
        raise KeyError(f"target_encoding recipe '{recipe.name}' is missing 'cell_means' " f"or 'factorize_lookup' in extra. Re-fit to materialize.")
    name_a, name_b = recipe.src_names
    nbins_a, nbins_b = recipe.factorize_nbins
    factorize_lookup: np.ndarray = recipe.extra["factorize_lookup"]
    cell_means: np.ndarray = recipe.extra["cell_means"]
    global_mean: float = recipe.extra["global_mean"]

    # ``cat_code_maps`` reproduces fit-time categorical / string codes (see _apply_factorize); ``src_bin_edges``
    # reproduces fit-time quantile codes for include_numeric numeric sources; numeric sources without edges fall
    # through to the int-cast path.
    _cat_maps = recipe.extra.get("cat_code_maps") or {}
    _edges = recipe.extra.get("src_bin_edges") or {}
    vals_a = _coerce_to_int_with_nan_handling(
        _extract_column(X, name_a, col_cache=col_cache), nbins_a, recipe.name, name_a, recipe.unknown_strategy,
        _cat_maps.get(name_a), _edges.get(name_a),
    )
    vals_b = _coerce_to_int_with_nan_handling(
        _extract_column(X, name_b, col_cache=col_cache), nbins_b, recipe.name, name_b, recipe.unknown_strategy,
        _cat_maps.get(name_b), _edges.get(name_b),
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
