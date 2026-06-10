"""Linear-residual + logratio composite transforms carved out of
``mlframe.training.composite_transforms``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _linear_residual_fit``
resolves transparently.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple,
)

import numpy as np

try:
    import numba as _numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _numba = None  # type: ignore
    _HAS_NUMBA = False

logger = logging.getLogger("mlframe.training.composite_transforms")

# Parent-resident constants referenced as default-arg values in function
# signatures below. Function-signature defaults evaluate at module load, so
# these MUST be top-level (lazy import inside the body wouldn't see them).
# The parent defines all four BEFORE its bottom-of-module sibling import,
# so this static cycle resolves at runtime. Whitelisted in
# tests/test_meta/test_no_import_cycles.py.
from . import (  # noqa: E402
    _GROUPED_MIN_GROUP_SIZE,
    _LINRES_ROBUST_MAD_K,
    _LINRES_ROBUST_MIN_KEEP_FRAC,
    _MULTI_BASE_COND_NUMBER_MAX,
)


logger = logging.getLogger("mlframe.training.composite_transforms")

def _logratio_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    # T_train computed in the valid domain (caller has already filtered).
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _MAD_FLOOR_FRAC, _MAD_SOFT_CAP_K
    t_train = np.log(y) - np.log(base)
    median_t = float(np.median(t_train))
    mad_train = float(np.median(np.abs(t_train - median_t)))
    # Floor against degenerate T_train (constant on train) -- otherwise
    # MAD = 0 collapses every prediction to ``base * exp(median_t)``,
    # which still ranks as "naive baseline" at predict time but at
    # least does not distort in-distribution predictions.
    std_y = float(np.std(y))
    mad_floor = _MAD_FLOOR_FRAC * std_y if std_y > 0 else 1e-6
    mad_eff = max(mad_train, mad_floor)
    return {
        "median_t": median_t,
        "mad_train": mad_train,
        "mad_eff": mad_eff,
        "soft_cap_k": _MAD_SOFT_CAP_K,
    }
def _logratio_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.log(y) - np.log(base)
def _logratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    median_t = float(params["median_t"])
    mad = float(params["mad_eff"])
    k = float(params["soft_cap_k"])
    # Soft-cap is centred on median(T_train), NOT on zero -- otherwise
    # any T distribution offset from zero (the typical case for
    # logratio when y and base have similar scale) gets clobbered by
    # the cap and inverse predictions collapse to ``base``.
    cap = k * mad
    t_capped = np.clip(t_hat, median_t - cap, median_t + cap)
    return base * np.exp(t_capped)
def _logratio_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (base > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y) & (y > 0)
def _linear_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """OLS fit with optional sample weights.

    Weighted least squares is implemented in closed form via
    ``np.linalg.lstsq`` on the row-scaled system
    ``sqrt(w) * X * beta = sqrt(w) * y`` (standard reformulation).
    Weights are normalised to sum to ``len(y)`` so the fit's
    numerical scale matches the unweighted case (avoids tiny
    coefficients on small w values).
    """
    n = len(y)
    if n < 2:
        return {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}
    X = np.column_stack([base.astype(np.float64), np.ones(n, dtype=np.float64)])
    y_f = y.astype(np.float64)

    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"_linear_residual_fit: sample_weight length {w.size} != y length {n}"
            )
        # Drop non-positive weights silently; warn if all zero.
        finite = np.isfinite(w) & (w > 0)
        if not finite.any():
            return {"alpha": 0.0, "beta": float(np.mean(y_f))}
        # Normalise weights to mean 1 so the system has the same
        # numerical scale as the unweighted version. lstsq handles
        # rank-deficient cases.
        w_norm = w[finite]
        w_norm = w_norm * (n / w_norm.sum())
        sw = np.sqrt(w_norm)
        X_w = X[finite] * sw[:, None]
        y_w = y_f[finite] * sw
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    return {"alpha": alpha, "beta": beta}
def _linear_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return y - alpha * base - beta
def _linear_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return t_hat + alpha * base + beta
def _linear_residual_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)
def _linear_residual_robust_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Trimmed-LS fit: OLS -> drop |resid|>3*MAD -> refit OLS.

    Returns the same ``{"alpha", "beta"}`` dict as :func:`_linear_residual_fit` so the existing forward / inverse functions work unchanged. ``sample_weight`` is honoured in BOTH passes.

    When the MAD-trim step doesn't drop any rows (no outliers above
    3*sigma_MAD), the second-pass OLS produces alpha/beta IDENTICAL to
    the first pass — the transform is mathematically equivalent to
    plain ``linear_residual``. We stamp ``is_redundant_with_linres=True``
    on the returned dict so composite discovery can skip the duplicate
    evaluation (observed in a prod log: y-linres-Y and
    y-linresR-Y produced identical RMSE=21.5433 because no outliers
    were trimmed -- pure duplicate compute).
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    n = len(y)
    if n < 2:
        result = {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}
        result["is_redundant_with_linres"] = True  # constant fit, OLS would do the same
        return result

    # Pass 1: standard OLS.
    first_pass = _linear_residual_fit(y, base, sample_weight=sample_weight)
    alpha1 = float(first_pass["alpha"])
    beta1 = float(first_pass["beta"])

    # Residuals + robust scale via MAD (sigma-equivalent multiplier 1.4826).
    resid = y.astype(np.float64) - alpha1 * base.astype(np.float64) - beta1
    if not np.all(np.isfinite(resid)):
        first_pass["is_redundant_with_linres"] = True
        return first_pass
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_mad = mad * 1.4826
    if sigma_mad <= 0.0 or not np.isfinite(sigma_mad):
        # Constant residual or numerical pathology -- OLS already covers it.
        first_pass["is_redundant_with_linres"] = True
        return first_pass

    keep = np.abs(resid - med) <= _LINRES_ROBUST_MAD_K * sigma_mad
    n_kept = int(keep.sum())
    if n_kept < max(2, int(_LINRES_ROBUST_MIN_KEEP_FRAC * n)):
        first_pass["is_redundant_with_linres"] = True
        return first_pass
    if n_kept == n:
        # No outliers trimmed -> pass-2 OLS on inlier set IS pass-1 OLS.
        # Skip the redundant refit and mark for discovery-side dedup.
        first_pass["is_redundant_with_linres"] = True
        return first_pass

    # Pass 2: OLS on the inlier set.
    sw2 = None if sample_weight is None else np.asarray(sample_weight)[keep]
    return _linear_residual_fit(y[keep], base[keep], sample_weight=sw2)
def _linear_residual_multi_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Joint OLS fit for ``T = y - Σⱼ αⱼ·baseⱼ - β``.

    Parameters
    ----------
    y
        Training target (``shape=(n,)``).
    base
        Either 1-D ``(n,)`` (K=1, equivalent to single-base linear_residual)
        or 2-D ``(n, K)`` for K-base joint fit.
    sample_weight
        Optional row weights - same weighted-LS reformulation as
        single-base ``_linear_residual_fit``.

    Returns
    -------
    dict with keys:
    - ``alphas``: list of K floats (one per base column).
    - ``beta``: float intercept.
    - ``condition_number``: float - design-matrix condition number;
      diagnostic only.
    - ``collinear_fallback``: bool - True if condition number > the
      gate, in which case ``alphas`` is all-zero and ``beta`` is the
      train mean of ``y`` (transform degenerates to ``T = y - mean(y)``).
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base_f = base.astype(np.float64)
    y_f = y.astype(np.float64)
    # Drop non-finite rows (non-finite y OR any non-finite base column) before
    # the OLS. Lag / rolling bases carry leading NaN; np.linalg.lstsq / svd
    # raise LinAlgError on NaN, which the callers' broad except silently
    # degrades to a single-base / collinear (alphas=0) fallback -- so the
    # default-ON multi-base promotion died precisely on the canonical lag-base
    # case, losing the benchmark-validated win. Masking here makes every caller
    # NaN-safe and mirrors _linear_residual_multi_domain's finite gate.
    row_finite = np.isfinite(y_f) & np.all(np.isfinite(base_f), axis=1)
    if not bool(row_finite.all()):
        base_f = base_f[row_finite]
        y_f = y_f[row_finite]
        if sample_weight is not None:
            sample_weight = np.asarray(
                sample_weight, dtype=np.float64,
            ).reshape(-1)[row_finite]
    n, k = base_f.shape
    if n < k + 1:
        return {
            "alphas": [0.0] * k,
            "beta": float(np.mean(y_f)) if n > 0 else 0.0,
            "condition_number": float("nan"),
            "collinear_fallback": True,
        }
    X = np.column_stack([base_f, np.ones(n, dtype=np.float64)])

    # Condition-number gate (Belsley/Kuh/Welsch). Computed on the
    # CENTERED base columns ONLY: including the intercept column would
    # conflate uncentered scale (e.g. ``base ~ N(10, 2)`` gives cond
    # >> 30 just from the offset, not from real multicollinearity).
    # For K=1 the centered base is just a single column => cond = 1
    # by construction (still go through the path so the code shape
    # matches the K>1 branch).
    try:
        if k == 1:
            cond = 1.0
        else:
            base_centered = base_f - base_f.mean(axis=0, keepdims=True)
            # Guard against an exactly-constant base column (zero
            # singular value); cond is infinite by definition there.
            col_norms = np.linalg.norm(base_centered, axis=0)
            if np.any(col_norms < 1e-12):
                cond = float("inf")
            else:
                # BKW scaled condition index: equilibrate columns to unit
                # norm BEFORE the SVD. Without this the condition number is
                # scale-VARIANT -- two independent bases in different units
                # (e.g. N(0,1) and N(0,1e6)) read cond ~ 1e6 and get falsely
                # rejected as collinear (alphas=0), silently killing a valid
                # multi-base residual. Unit-norm scaling measures genuine
                # multicollinearity (column angle), not unit mismatch.
                base_scaled = base_centered / col_norms
                sv = np.linalg.svd(base_scaled, compute_uv=False)
                cond = float(sv.max() / max(sv.min(), np.finfo(np.float64).tiny))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if cond > _MULTI_BASE_COND_NUMBER_MAX or not np.isfinite(cond):
        return {
            "alphas": [0.0] * k,
            "beta": float(np.mean(y_f)),
            "condition_number": cond,
            "collinear_fallback": True,
        }

    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"_linear_residual_multi_fit: sample_weight length {w.size} "
                f"!= y length {n}"
            )
        finite = np.isfinite(w) & (w > 0)
        if not finite.any():
            return {
                "alphas": [0.0] * k,
                "beta": float(np.mean(y_f)),
                "condition_number": cond,
                "collinear_fallback": True,
            }
        w_norm = w[finite] * (n / w[finite].sum())
        sw = np.sqrt(w_norm)
        X_w = X[finite] * sw[:, None]
        y_w = y_f[finite] * sw
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    return {
        "alphas": [float(c) for c in coef[:k]],
        "beta": float(coef[-1]),
        "condition_number": cond,
        "collinear_fallback": False,
    }
def _linear_residual_multi_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alphas = np.asarray(params["alphas"], dtype=np.float64)
    beta = float(params["beta"])
    if base.shape[1] != alphas.size:
        raise ValueError(
            f"linear_residual_multi: base has {base.shape[1]} columns but "
            f"fitted alphas has {alphas.size} entries"
        )
    return y - (base.astype(np.float64) @ alphas) - beta
def _linear_residual_multi_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alphas = np.asarray(params["alphas"], dtype=np.float64)
    beta = float(params["beta"])
    if base.shape[1] != alphas.size:
        raise ValueError(
            f"linear_residual_multi: base has {base.shape[1]} columns but "
            f"fitted alphas has {alphas.size} entries"
        )
    return t_hat + (base.astype(np.float64) @ alphas) + beta
def _linear_residual_multi_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base_ok = np.all(np.isfinite(base), axis=1)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)
def _linear_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    min_group_size: int = _GROUPED_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Per-group OLS fit with James-Stein shrinkage toward global α.

    Parameters
    ----------
    y, base
        Training target and base (1-D arrays, length n).
    groups
        Per-row group labels (1-D ndarray, length n). Required; raised
        as ValueError if None.
    sample_weight
        Optional row weights (currently unused for the per-group OLS
        because weight semantics within small groups are unstable; the
        global fit honours weights via standard OLS).
    min_group_size
        Rows-per-group threshold below which a group skips its own
        OLS and uses the global α/β.

    Returns
    -------
    fitted_params with keys:
    - ``alpha_global`` / ``beta_global``: global OLS fit on all rows.
    - ``per_group_alphas`` / ``per_group_betas``: dict[group_label ->
      float]. Keys are str(group_label) for JSON-serialisability.
    - ``shrinkage_factor``: float in [0, 1] from James-Stein.
    - ``group_sizes``: dict[str(group_label) -> int]; diagnostic.
    - ``min_group_size``: int; the threshold used.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _GROUPED_MIN_GROUP_SIZE, _james_stein_shrinkage_factor
    if groups is None:
        raise ValueError(
            "linear_residual_grouped requires a 1-D ``groups`` array of "
            "per-row group labels (configure ``group_column`` on the "
            "wrapper). For ungrouped fit, use ``linear_residual``."
        )
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(y):
        raise ValueError(
            f"linear_residual_grouped: groups has {len(groups)} rows but "
            f"y has {len(y)} rows."
        )

    # Global OLS: same logic as legacy linear_residual.
    global_params = _linear_residual_fit(y, base, sample_weight=sample_weight)
    alpha_global = float(global_params["alpha"])
    beta_global = float(global_params["beta"])

    # Per-group OLS.
    per_group_alphas: dict[str, float] = {}
    per_group_betas: dict[str, float] = {}
    group_sizes: dict[str, int] = {}
    unique_groups, inverse_idx = np.unique(groups, return_inverse=True)

    # Cache residual squared sum across all groups to estimate σ² for
    # James-Stein. Computed against the per-group OLS predictions (the
    # estimator we're trying to shrink).
    total_resid_sq = 0.0
    total_n = 0

    alphas_for_shrink: list[float] = []
    sizes_for_shrink: list[float] = []

    for i, g in enumerate(unique_groups):
        g_mask = (inverse_idx == i)
        n_g = int(g_mask.sum())
        g_key = str(g)
        group_sizes[g_key] = n_g
        if n_g < min_group_size:
            # Skip per-group OLS; defer to global.
            per_group_alphas[g_key] = alpha_global
            per_group_betas[g_key] = beta_global
            continue
        y_g = y[g_mask]
        base_g = base[g_mask]
        sw_g = sample_weight[g_mask] if sample_weight is not None else None
        try:
            params_g = _linear_residual_fit(y_g, base_g, sample_weight=sw_g)
            a_g = float(params_g["alpha"])
            b_g = float(params_g["beta"])
        except Exception:  # pragma: no cover - defensive
            a_g, b_g = alpha_global, beta_global
        per_group_alphas[g_key] = a_g
        per_group_betas[g_key] = b_g
        alphas_for_shrink.append(a_g)
        sizes_for_shrink.append(float(n_g))
        # Accumulate residuals for σ² estimate.
        resid = y_g - a_g * base_g - b_g
        total_resid_sq += float(np.sum(resid * resid))
        total_n += n_g

    # James-Stein shrinkage of the eligible per-group alphas.
    if len(alphas_for_shrink) >= 4 and total_n > 0:
        sigma2 = total_resid_sq / max(total_n - 2 * len(alphas_for_shrink), 1)
        c = _james_stein_shrinkage_factor(
            np.asarray(alphas_for_shrink, dtype=np.float64),
            alpha_global,
            np.asarray(sizes_for_shrink, dtype=np.float64),
            sigma2,
        )
    else:
        c = 0.0
    # Apply shrinkage to eligible groups (ones that ran their own OLS).
    if c > 0:
        for g_key, a_g in list(per_group_alphas.items()):
            n_g = group_sizes[g_key]
            if n_g < min_group_size:
                continue
            per_group_alphas[g_key] = (1.0 - c) * a_g + c * alpha_global

    return {
        "alpha_global": alpha_global,
        "beta_global": beta_global,
        "per_group_alphas": per_group_alphas,
        "per_group_betas": per_group_betas,
        "shrinkage_factor": float(c),
        "group_sizes": group_sizes,
        "min_group_size": int(min_group_size),
    }
def _linear_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _row_alpha_beta
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.forward: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return y - row_alpha * base - row_beta
def _linear_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _row_alpha_beta
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.inverse: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return t_hat + row_alpha * base + row_beta
def _linear_residual_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    return _linear_residual_domain(y, base)
