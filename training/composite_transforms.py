"""Composite-target transform registry: 11 transforms (diff, ratio, logratio, linear_residual + multi/grouped/quantile/monotonic/ewma/rolling_quantile/frac_diff extended set) + Transform dataclass + UnknownTransformError / DomainViolationError. Split out of composite.py to keep transform-implementation surface separate from the wrapper + discovery surface; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# Soft-cap MAD floor: when MAD(T_train) is below
# ``_MAD_FLOOR_FRAC * std(y_train)``, we substitute the latter to keep
# the soft-cap bound numerically meaningful even if the transform
# produced a degenerate (near-constant) T on train. Without this,
# logratio's MAD-cap collapses to zero on degenerate train and every
# prediction inverts to ``base * exp(0) = base`` silently.
_MAD_FLOOR_FRAC: float = 1e-3

# Multiplier for MAD-soft-cap on T_hat (logratio in particular).
_MAD_SOFT_CAP_K: float = 10.0


class UnknownTransformError(KeyError):
    """Raised when a transform name is not in :data:`_TRANSFORMS_REGISTRY`."""


class DomainViolationError(ValueError):
    """Raised at fit time when the input domain is incompatible with the
    chosen transform (e.g. ``logratio`` requested but ``y`` contains
    non-positive values).

    At predict time we do NOT raise -- per-row violations are handled
    via fall-back values + counters logged on
    ``CompositeTargetEstimator.runtime_stats_``.
    """

# ----------------------------------------------------------------------
# Transform registry
# ----------------------------------------------------------------------

# Tags used to filter the registry into presets.
TAG_CORE: str = "core"           # diff / ratio / logratio / linear_residual
TAG_EXTENDED: str = "extended"   # placeholder; future presets may add more
TAG_REGRESSION: str = "regression"


@dataclass(frozen=True)
class Transform:
    """One row of the transform registry.

    The four functions form a contract:

    - ``fit(y_train, base_train)`` -> ``dict`` of transform-specific
      fitted parameters (e.g. ``{"alpha": float, "beta": float}``).
      Pure: must NOT mutate inputs and must NOT close over external
      state. The dict is JSON-serialisable.
    - ``forward(y, base, params)`` -> ``T``: applies the transform.
    - ``inverse(T_hat, base, params)`` -> ``y_hat``: applies the
      inverse. Wrapper additionally clips the output to the y-bounds
      stored alongside ``params``.
    - ``domain_check(y, base)`` -> boolean mask of valid rows. Wrapper
      uses this at fit-time to drop invalid rows BEFORE calling
      ``fit`` / ``forward``, and at predict-time to flag rows where
      the inverse cannot be applied cleanly (those rows fall back to
      ``y_train_median``).
    """

    name: str
    forward: Callable[..., np.ndarray]
    inverse: Callable[..., np.ndarray]
    fit: Callable[..., Dict[str, Any]]
    domain_check: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str
    tags: FrozenSet[str] = field(default_factory=frozenset)
    # R10c #3 (2026-05-11): grouped-transform support. When True, the
    # wrapper extracts a ``groups`` array from a configured column and
    # passes it as a keyword argument to ``fit`` / ``forward`` /
    # ``inverse``. Used by ``linear_residual_grouped`` (per-segment alpha
    # with James-Stein shrinkage). Default False: legacy transforms
    # keep their 3-arg signatures and the wrapper never passes groups.
    requires_groups: bool = False


# ----------------------------------------------------------------------
# diff: T = y - base. Always defined, no params, no domain restrictions.
# ----------------------------------------------------------------------

def _diff_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    return {}


def _diff_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return y - base


def _diff_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return t_hat + base


def _diff_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# ratio: T = y / base. Requires |base| >= eps.
# ----------------------------------------------------------------------

def _ratio_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    # eps relative to the typical scale of base on train -- small enough
    # not to bias the transform but large enough to keep division
    # numerically clean. Stored in params so predict time uses the
    # SAME eps (no train/test drift).
    scale = float(np.median(np.abs(base[np.isfinite(base) & (base != 0)])))
    eps = max(scale * 1e-6, 1e-12) if scale > 0 else 1e-12
    return {"eps": eps}


def _ratio_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return y / safe_base


def _ratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return t_hat * base


def _ratio_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (np.abs(base) > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# logratio: T = log(y) - log(base). Requires y, base > 0.
# Inverse uses MAD-soft-cap on T_hat to prevent exp() blow-up.
# ----------------------------------------------------------------------

def _logratio_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    # T_train computed in the valid domain (caller has already filtered).
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


def _logratio_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return np.log(y) - np.log(base)


def _logratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
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


def _logratio_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (base > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y) & (y > 0)


# ----------------------------------------------------------------------
# linear_residual: T = y - alpha*base - beta. OLS on train.
# ----------------------------------------------------------------------

def _linear_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
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
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return y - alpha * base - beta


def _linear_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return t_hat + alpha * base + beta


def _linear_residual_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# linear_residual_multi (#1 from R10c brainstorm).
# T = y - Σⱼ αⱼ·baseⱼ - β. Joint OLS on (n, K) base matrix.
#
# Same inverse contract as single-base linear_residual; the only thing
# that changes is the shape of ``base``: 2-D ``(n, K)`` instead of 1-D
# ``(n,)``. Callers materialise the matrix from ``spec.base_columns``
# at predict time. ``fit`` accepts 1-D or 2-D ``base`` for ergonomics
# (1-D is degenerate-K=1 multi-base, equivalent to plain
# ``linear_residual``).
#
# Joint OLS guard: when K bases are near-collinear (e.g. ``TVT_prev``
# and ``TVT_prev_smoothed`` together), the design-matrix condition
# number explodes and αs become unstable. We compute condition number
# upfront and reject (returning the zero-alpha fallback) above a
# configurable threshold (default 30) so the spec downstream gets a
# safe identity-like inverse rather than blowing up at predict time.
# ----------------------------------------------------------------------

# Condition-number gate above which joint OLS is considered unstable
# and the multi-base transform falls back to zero-alpha + intercept.
# 30 is the conventional threshold (Belsley/Kuh/Welsch); above that,
# multicollinearity inflates standard errors enough that the alpha
# estimates carry no useful information. Exposed as module-level so
# tests can monkey-patch without recompiling.
_MULTI_BASE_COND_NUMBER_MAX: float = 30.0


def _linear_residual_multi_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Joint OLS fit for ``T = y - Σⱼ αⱼ·baseⱼ - β``.

    Parameters
    ----------
    y
        Training target (``shape=(n,)``).
    base
        Either 1-D ``(n,)`` (K=1, equivalent to single-base linear_residual)
        or 2-D ``(n, K)`` for K-base joint fit.
    sample_weight
        Optional row weights — same weighted-LS reformulation as
        single-base ``_linear_residual_fit``.

    Returns
    -------
    dict with keys:
    - ``alphas``: list of K floats (one per base column).
    - ``beta``: float intercept.
    - ``condition_number``: float — design-matrix condition number;
      diagnostic only.
    - ``collinear_fallback``: bool — True if condition number > the
      gate, in which case ``alphas`` is all-zero and ``beta`` is the
      train mean of ``y`` (transform degenerates to ``T = y - mean(y)``).
    """
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    n, k = base.shape
    y_f = y.astype(np.float64)
    if n < k + 1:
        return {
            "alphas": [0.0] * k,
            "beta": float(np.mean(y_f)) if n > 0 else 0.0,
            "condition_number": float("nan"),
            "collinear_fallback": True,
        }
    base_f = base.astype(np.float64)
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
                sv = np.linalg.svd(base_centered, compute_uv=False)
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
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
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
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
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
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base_ok = np.all(np.isfinite(base), axis=1)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# linear_residual_grouped (#3 from R10c brainstorm).
#
# T = y - alpha_g * base - beta_g, where g = group(row). Alphas are fit
# per group via OLS, then shrunk toward the global alpha via empirical-
# Bayes (James-Stein style). The shrinkage protects small-n groups
# from over-fitted alphas while still letting large-n groups follow
# their own distinct AR dynamics.
#
# Use case (TVT): per-well α. Wells differ in autoregressive strength;
# one-α-across-wells is the wrong model. Per-well α + shrinkage handles
# the 5-row wells safely without losing the per-well structure that a
# global fit averages out.
#
# Failure handling:
# - At fit: small groups (n < min_group_size) skip OLS and use the
#   global α (i.e. fully shrunk to global).
# - At predict: unseen group IDs fall back to the global α.
# Both paths are silent + tracked in fitted_params for diagnostics.
# ----------------------------------------------------------------------

# Minimum rows per group below which the group falls back to the global
# fit. Belsley/Kuh/Welsch rule of thumb is "≥10× predictors" for stable
# OLS; for 1-base + intercept that's 20-30. We use 30 as a safe default;
# exposed as module-level so tests can monkey-patch.
_GROUPED_MIN_GROUP_SIZE: int = 30


def _james_stein_shrinkage_factor(
    per_group_alphas: np.ndarray,
    global_alpha: float,
    group_sizes: np.ndarray,
    sigma2_total: float,
) -> float:
    """Estimate the James-Stein shrinkage factor toward ``global_alpha``.

    Returns a scalar c ∈ [0, 1]:
    - c = 0: keep per-group alphas as-is (no shrinkage).
    - c = 1: collapse all per-group alphas to global_alpha (full shrinkage).

    The classic JS estimator for K group means with known variance σ² is

        c = max(0, 1 - (K - 3) σ² / Σ_g (α_g - global)² )

    We use the variance of residuals as σ² proxy (σ²/n_g per group),
    weighted by ``group_sizes``. When the per-group spread is large
    relative to noise, c -> 0 (let the data speak). When noise dominates
    spread, c -> 1 (shrink heavily).

    A degenerate case (K < 4 groups, or all alphas equal) returns c = 0
    so the JS correction can't reduce K below the JS-applicability
    threshold; the per-group estimates pass through unmodified.
    """
    k = per_group_alphas.size
    if k < 4:
        return 0.0
    deviations = per_group_alphas - global_alpha
    sum_sq = float(np.sum(deviations * deviations))
    if sum_sq <= 0:
        return 0.0
    # σ²_per_group ≈ σ²_total / mean(n_g) (residual variance per group).
    # Use mean per-group variance as the JS noise proxy.
    mean_per_group_variance = float(sigma2_total / max(np.mean(group_sizes), 1.0))
    # Classic JS shrinkage factor c in
    #     α_shrunk = (1-c) α_g + c α_global
    # c = (K-3) σ² / Σ_g (α_g - α_global)², clamped to [0, 1].
    # High noise / low spread => c -> 1 (full shrink to global).
    # Low noise / high spread  => c -> 0 (keep per-group alphas).
    raw = (k - 3) * mean_per_group_variance / sum_sq
    return float(max(0.0, min(1.0, raw)))


def _linear_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray,
    groups: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    min_group_size: int = _GROUPED_MIN_GROUP_SIZE,
) -> Dict[str, Any]:
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
    per_group_alphas: Dict[str, float] = {}
    per_group_betas: Dict[str, float] = {}
    group_sizes: Dict[str, int] = {}
    unique_groups, inverse_idx = np.unique(groups, return_inverse=True)

    # Cache residual squared sum across all groups to estimate σ² for
    # James-Stein. Computed against the per-group OLS predictions (the
    # estimator we're trying to shrink).
    total_resid_sq = 0.0
    total_n = 0

    alphas_for_shrink: List[float] = []
    sizes_for_shrink: List[float] = []

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


def _row_alpha_beta(
    groups: np.ndarray, params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise per-row (alpha, beta) from the grouped params dict.

    Vectorised: ``np.unique`` collapses the n-row groups vector to K
    unique labels, looks them up in the params dict ONCE per unique
    label, then uses inverse-indexing to broadcast back to n rows.
    A naive ``for i, g in enumerate(groups)`` is ~30x slower on 200K
    rows; cProfile measured the loop at 88% of total fit+predict cost
    pre-optimisation.

    Unseen group labels (present at predict but not at fit) fall back
    to global alpha/beta -- a safe identity-like inverse.
    """
    alpha_global = float(params["alpha_global"])
    beta_global = float(params["beta_global"])
    pg_alphas = params["per_group_alphas"]
    pg_betas = params["per_group_betas"]
    # K unique labels; inv maps each row to an index into uniq.
    uniq, inv = np.unique(groups, return_inverse=True)
    # Build per-unique-label alpha / beta with global as fallback.
    uniq_alpha = np.array(
        [pg_alphas.get(str(g), alpha_global) for g in uniq],
        dtype=np.float64,
    )
    uniq_beta = np.array(
        [pg_betas.get(str(g), beta_global) for g in uniq],
        dtype=np.float64,
    )
    return uniq_alpha[inv], uniq_beta[inv]


def _linear_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.forward: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return y - row_alpha * base - row_beta


def _linear_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.inverse: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return t_hat + row_alpha * base + row_beta


def _linear_residual_grouped_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    return _linear_residual_domain(y, base)


# ----------------------------------------------------------------------
# quantile_residual (#6 from R10c brainstorm).
#
# Non-parametric, heteroscedasticity-aware residual. Bin ``base`` into ``n_bins`` quantile buckets on train; for each bucket compute median(y) and IQR(y). T = (y - median_bin) / IQR_bin. Inverse: y_hat = T_hat * IQR_bin + median_bin. The bucket-conditional centring + scaling makes T near-iid (mean 0, scale 1) regardless of how Var(y|base) depends on base -- which linear_residual cannot do.
#
# Use case (TVT): when Var(TVT|TVT_prev) scales with TVT_prev (heteroscedastic, common in flow / queue / well-log data), conditional-quantile residual produces a near-iid target which tree models predict more cleanly than a heteroscedastic linear residual.
#
# Failure handling:
# - Bins with < ``min_bin_n`` train rows reuse the GLOBAL median(y) / IQR(y) so under-populated bins don't carry noisy bin-local estimates. Tracked in ``params["bin_sizes"]`` for diagnostics.
# - At predict, ``base`` values outside the train range fall back to the EDGE bin (first or last), not a separate "out-of-range" bucket. Consistent with the quantile-binning contract.
# - IQR == 0 for a bin (constant y) is replaced with the global IQR or a small floor (1e-6 * std(y_train)) to keep the inverse well-defined.
# ----------------------------------------------------------------------

_QUANTILE_RESIDUAL_DEFAULT_N_BINS: int = 10
_QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N: int = 50


def _quantile_residual_fit(
    y: np.ndarray, base: np.ndarray,
    n_bins: int = _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    min_bin_n: int = _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
) -> Dict[str, Any]:
    """Fit per-bucket median(y) + IQR(y) over ``n_bins`` quantile bins of ``base``.

    Returns
    -------
    dict with keys:
    - ``bin_edges``: 1-D ndarray of length ``n_bins+1`` (open at -inf, +inf).
    - ``bin_medians``: 1-D ndarray of length ``n_bins`` (median(y) per bin; global median for under-populated bins).
    - ``bin_iqrs``: 1-D ndarray of length ``n_bins`` (IQR(y) per bin; global IQR with floor for under-populated / constant bins).
    - ``bin_sizes``: list[int] of length ``n_bins`` (train rows per bin).
    - ``global_median``: float (median of train y, used as fallback).
    - ``global_iqr``: float (IQR of train y, used as fallback).
    - ``n_bins``: int (recorded for predict-time validation).
    """
    n_bins = max(2, int(n_bins))
    min_bin_n = max(2, int(min_bin_n))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if finite.sum() < n_bins * 2:
        # Degenerate: fall back to global stats so the inverse is still safe.
        med = float(np.median(y_f[finite])) if finite.any() else 0.0
        iqr_v = float(np.subtract(*np.percentile(y_f[finite], [75, 25]))) if finite.sum() >= 4 else 1.0
        iqr_v = max(iqr_v, 1e-6)
        return {
            "bin_edges": np.array([-np.inf, np.inf], dtype=np.float64),
            "bin_medians": np.array([med], dtype=np.float64),
            "bin_iqrs": np.array([iqr_v], dtype=np.float64),
            "bin_sizes": [int(finite.sum())],
            "global_median": med,
            "global_iqr": iqr_v,
            "n_bins": 1,
        }
    y_clean = y_f[finite]
    base_clean = base_f[finite]
    # Quantile edges on train base. ``np.quantile`` with linspace covers the open-open envelope; we replace the outermost edges with +/-inf so the predict-time digitize never produces an out-of-range bucket.
    inner_qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(base_clean, inner_qs)
    # Deduplicate edges (when many ties at one quantile, several edges collapse) -- empty bins would otherwise emerge. Tolerate up to n_bins-1 unique edges; clip n_bins accordingly downstream.
    edges = np.unique(edges)
    if edges.size < 2:
        # All base values identical: degenerate single bucket.
        med = float(np.median(y_clean))
        iqr_v = max(float(np.subtract(*np.percentile(y_clean, [75, 25]))), 1e-6)
        return {
            "bin_edges": np.array([-np.inf, np.inf], dtype=np.float64),
            "bin_medians": np.array([med], dtype=np.float64),
            "bin_iqrs": np.array([iqr_v], dtype=np.float64),
            "bin_sizes": [int(y_clean.size)],
            "global_median": med,
            "global_iqr": iqr_v,
            "n_bins": 1,
        }
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_n_bins = edges.size - 1
    # Global stats (fallback for under-populated bins).
    global_median = float(np.median(y_clean))
    global_iqr = max(float(np.subtract(*np.percentile(y_clean, [75, 25]))), 1e-6)
    # Per-bin assignment via np.searchsorted (right-side: edges[i-1] <= x < edges[i]).
    bin_idx = np.clip(np.searchsorted(edges[1:-1], base_clean, side="right"), 0, actual_n_bins - 1)
    bin_medians = np.full(actual_n_bins, global_median, dtype=np.float64)
    bin_iqrs = np.full(actual_n_bins, global_iqr, dtype=np.float64)
    bin_sizes: List[int] = []
    for b in range(actual_n_bins):
        mask = bin_idx == b
        bin_n = int(mask.sum())
        bin_sizes.append(bin_n)
        if bin_n < min_bin_n:
            # Under-populated: keep global fallback.
            continue
        bin_y = y_clean[mask]
        bin_medians[b] = float(np.median(bin_y))
        bin_iqr = float(np.subtract(*np.percentile(bin_y, [75, 25])))
        # Floor IQR against constant-y bins so the inverse stays well-defined; use global IQR as a sensible scale anchor rather than 1e-6.
        bin_iqrs[b] = bin_iqr if bin_iqr > 1e-6 else global_iqr
    return {
        "bin_edges": edges,
        "bin_medians": bin_medians,
        "bin_iqrs": bin_iqrs,
        "bin_sizes": bin_sizes,
        "global_median": global_median,
        "global_iqr": global_iqr,
        "n_bins": int(actual_n_bins),
    }


def _quantile_residual_assign_bins(base: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each row of ``base`` to a bin index in [0, n_bins-1]. Out-of-range values map to the edge bin (no separate OOR bucket), per the contract documented on the transform."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    # ``edges[1:-1]`` are the INNER cut points; searchsorted returns 0..n_bins.
    bin_idx = np.searchsorted(edges[1:-1], base_f, side="right")
    return np.clip(bin_idx, 0, n_bins - 1)


def _quantile_residual_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return (np.asarray(y, dtype=np.float64) - medians[bin_idx]) / iqrs[bin_idx]


def _quantile_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return np.asarray(t_hat, dtype=np.float64) * iqrs[bin_idx] + medians[bin_idx]


def _quantile_residual_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# monotonic_residual (R10c brainstorm round-2 extension A; non-parametric monotonic-spline residual).
#
# T = y - g(base), where g is a monotonic PCHIP interpolant fitted to per-knot median(y) on quantile-based knots of base. Generalises linear_residual to capture saturating / sigmoidal / convex-concave relationships that the linear OLS leaves in the residual.
#
# Use case (TVT-style well-log): if TVT ~= a*TVT_prev grows linearly at low values but plateaus at high values, linear_residual leaves a wedge of curvature in T. Monotonic-spline residual sucks up the curvature so the inner model sees a near-iid T.
#
# PCHIP (Piecewise Cubic Hermite Interpolating Polynomial; scipy) is monotone-preserving by construction -- the interpolant between two adjacent knots is monotone if the knot values are monotone. We force per-knot y-values to be monotone (cumulative max or cumulative min depending on the train-data slope) so the spline g is monotone overall. This regularises the fit against noise-driven non-monotonicities at small per-knot sample sizes.
# ----------------------------------------------------------------------

_MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS: int = 12
_MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N: int = 30


def _monotonic_residual_fit(
    y: np.ndarray, base: np.ndarray,
    n_knots: int = _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    min_knot_n: int = _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
) -> Dict[str, Any]:
    """Fit a monotone PCHIP spline g(base) via per-quantile-knot medians and orient by the sign of the global Spearman correlation between y and base. Stores the knot x/y arrays + the global y mean as a fallback. Domain at predict time: base values outside [knots_x[0], knots_x[-1]] are clipped to the edge knots (PCHIP extrapolation is not safe -- it can run off to +/- inf rapidly)."""
    n_knots = max(3, int(n_knots))
    min_knot_n = max(2, int(min_knot_n))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if finite.sum() < n_knots * 2:
        y_med = float(np.median(y_f[finite])) if finite.any() else 0.0
        return {
            "knots_x": np.array([0.0, 1.0], dtype=np.float64),
            "knots_y": np.array([y_med, y_med], dtype=np.float64),
            "y_train_mean": y_med,
            "monotone_direction": 0,
            "n_knots_effective": 2,
        }
    y_clean = y_f[finite]
    base_clean = base_f[finite]
    # Knot x positions on quantile cuts of base (NOT linearly-spaced — uneven base distributions benefit from quantile placement).
    qs = np.linspace(0.0, 1.0, n_knots)
    knots_x = np.quantile(base_clean, qs)
    # Deduplicate ties (many identical base values collapse to fewer knots).
    knots_x = np.unique(knots_x)
    if knots_x.size < 3:
        y_med = float(np.median(y_clean))
        return {
            "knots_x": np.array([base_clean.min(), base_clean.max()], dtype=np.float64),
            "knots_y": np.array([y_med, y_med], dtype=np.float64),
            "y_train_mean": y_med,
            "monotone_direction": 0,
            "n_knots_effective": 2,
        }
    # Per-knot y values: median(y) for rows assigned to each knot's quantile slab.
    n_eff = knots_x.size
    knots_y = np.empty(n_eff, dtype=np.float64)
    y_global_med = float(np.median(y_clean))
    # Slab boundaries: midpoints between adjacent knots (left/right edges extend to +/-inf so every row maps to a slab).
    slab_edges = np.empty(n_eff + 1, dtype=np.float64)
    slab_edges[0] = -np.inf
    slab_edges[-1] = np.inf
    slab_edges[1:-1] = 0.5 * (knots_x[:-1] + knots_x[1:])
    slab_idx = np.clip(np.searchsorted(slab_edges[1:-1], base_clean, side="right"), 0, n_eff - 1)
    for k in range(n_eff):
        mask = slab_idx == k
        n_in_slab = int(mask.sum())
        if n_in_slab < min_knot_n:
            knots_y[k] = y_global_med
        else:
            knots_y[k] = float(np.median(y_clean[mask]))
    # Orient monotonicity by Spearman correlation between y and base; tie -> increasing (arbitrary but stable).
    if y_clean.size >= 3 and base_clean.size >= 3:
        # np.corrcoef on ranks ~ Spearman; avoids scipy dep.
        from scipy.stats import spearmanr  # lazy import
        try:
            rho, _ = spearmanr(base_clean, y_clean)
            direction = 1 if (rho is None or not np.isfinite(rho) or rho >= 0) else -1
        except Exception:
            direction = 1
    else:
        direction = 1
    # Enforce monotonicity by cumulative max / min over knots in the orientation direction. This protects against per-knot median noise creating local non-monotonicities the PCHIP would otherwise honour (PCHIP is monotone PER SEGMENT but only if the knot values are monotone overall).
    if direction == 1:
        knots_y = np.maximum.accumulate(knots_y)
    else:
        knots_y = np.minimum.accumulate(knots_y)
    return {
        "knots_x": knots_x,
        "knots_y": knots_y,
        "y_train_mean": float(np.mean(y_clean)),
        "monotone_direction": direction,
        "n_knots_effective": int(n_eff),
    }


def _monotonic_residual_g(base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Evaluate the monotone PCHIP interpolant at the requested base values. Out-of-range values clip to the edge knot value (NOT extrapolated)."""
    knots_x = np.asarray(params["knots_x"], dtype=np.float64)
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    if knots_x.size < 2:
        return np.full(base_f.shape, float(params.get("y_train_mean", 0.0)), dtype=np.float64)
    if knots_x.size == 2:
        # Degenerate: linear interpolation between the two anchor knots; out-of-range clamps to edge value.
        clipped = np.clip(base_f, knots_x[0], knots_x[-1])
        slope = (knots_y[-1] - knots_y[0]) / max(knots_x[-1] - knots_x[0], 1e-12)
        return knots_y[0] + slope * (clipped - knots_x[0])
    from scipy.interpolate import PchipInterpolator  # lazy
    interp = PchipInterpolator(knots_x, knots_y, extrapolate=False)
    # extrapolate=False yields NaN outside [x[0], x[-1]]; fill those with the edge knot values to keep predict-time well-defined.
    out = interp(base_f)
    if np.any(~np.isfinite(out)):
        low_mask = base_f < knots_x[0]
        high_mask = base_f > knots_x[-1]
        out[low_mask] = knots_y[0]
        out[high_mask] = knots_y[-1]
    return out


def _monotonic_residual_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - _monotonic_residual_g(base, params)


def _monotonic_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    return np.asarray(t_hat, dtype=np.float64) + _monotonic_residual_g(base, params)


def _monotonic_residual_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# ewma_residual (R10c brainstorm #5a; exponentially-weighted moving average residual).
#
# T = y - EWMA_k(base), where EWMA_k(x_i) = (1 - alpha) * EWMA_k(x_{i-1}) + alpha * x_i with alpha = 2 / (k + 1) (standard half-life convention). Time-ordered: rows are assumed to be in chronological order (caller responsible). Inverse: y_hat = T_hat + EWMA_k(base), reusing the SAME EWMA recursion at predict time.
#
# Use case: lag-1 ``TVT_prev`` captures only one autoregressive horizon. EWMA(k=3..21) captures slow drift / regime persistence beyond a single lag. Adds transform diversity to the cross-target ensemble which NNLS exploits.
#
# Sequence contract: ``base`` must be in chronological order at both fit and predict. The transform DOES NOT receive a time-column kwarg; the caller (or wrapper-level orchestration) is responsible for ensuring row order matches time order. Out-of-order data produces a valid EWMA but with semantics the caller did not intend.
# ----------------------------------------------------------------------

_EWMA_RESIDUAL_DEFAULT_K: int = 7  # half-life-like span; alpha = 2 / (k + 1) ~= 0.25
_FRAC_DIFF_DEFAULT_D: float = 0.5  # Lopez de Prado standard order
_FRAC_DIFF_DEFAULT_LAGS: int = 30  # maximum weight tail used in the truncated series


def _ewma_residual_fit(
    y: np.ndarray, base: np.ndarray, k: int = _EWMA_RESIDUAL_DEFAULT_K,
) -> Dict[str, Any]:
    """Fit stores only the EWMA half-life span ``k``. The EWMA itself is re-computed at forward / inverse time -- this keeps the fitted params JSON-serialisable and stateless (the alternative of storing the full N-row EWMA trace would bloat metadata and break predict-on-new-data). The first-row anchor is the train-base mean: ``ewma[0] = mean(base_train)``."""
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(base_f)
    anchor = float(np.mean(base_f[finite])) if finite.any() else 0.0
    return {"k": k, "anchor": anchor}


def _ewma_compute(base: np.ndarray, k: int, anchor: float) -> np.ndarray:
    """Exponentially-weighted moving average using ``alpha = 2 / (k + 1)``. Non-finite base values inherit the previous EWMA state (carry-forward), which keeps the recursion well-defined on rows the upstream domain check did not yet flag."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    alpha = 2.0 / (k + 1)
    out = np.empty(base_f.size, dtype=np.float64)
    state = anchor
    for i in range(base_f.size):
        x = base_f[i]
        if np.isfinite(x):
            state = (1.0 - alpha) * state + alpha * x
        out[i] = state
    return out


def _ewma_residual_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - _ewma_compute(
        base, int(params["k"]), float(params["anchor"]),
    )


def _ewma_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    return np.asarray(t_hat, dtype=np.float64) + _ewma_compute(
        base, int(params["k"]), float(params["anchor"]),
    )


def _ewma_residual_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# rolling_quantile_ratio (R10c brainstorm #5b).
#
# T = y / max(RollingQ50_k(base), eps), where RollingQ50_k is the centred rolling median of ``base`` over a window of ``k`` rows. Inverse: y_hat = T_hat * RollingQ50_k(base). For the first ``k-1`` rows the window is left-truncated, falling back to the median of available rows.
#
# Use case: multiplicative DGP where the local-median of base sets the y-scale. Logratio captures global multiplicative structure but not local windowing. ``rolling_quantile_ratio`` is the localised version.
# ----------------------------------------------------------------------

_ROLLING_QUANTILE_DEFAULT_K: int = 7


def _rolling_quantile_ratio_fit(
    y: np.ndarray, base: np.ndarray, k: int = _ROLLING_QUANTILE_DEFAULT_K,
) -> Dict[str, Any]:
    """Stores the window span ``k`` and an eps floor derived from train base scale to keep division safe at predict time on near-zero rolling medians."""
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(base_f) & (base_f != 0)
    scale = float(np.median(np.abs(base_f[finite]))) if finite.any() else 1.0
    eps = max(scale * 1e-6, 1e-12)
    return {"k": k, "eps": eps}


def _rolling_median(arr: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling median with truncation at boundaries. O(n*k log k) which is acceptable for k in 3..21; not optimised for very large windows."""
    n = arr.size
    half = k // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = arr[lo:hi]
        finite = np.isfinite(window)
        if finite.any():
            out[i] = float(np.median(window[finite]))
        else:
            out[i] = arr[i] if np.isfinite(arr[i]) else 0.0
    return out


def _rolling_quantile_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    k = int(params["k"])
    eps = float(params["eps"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(y, dtype=np.float64) / safe


def _rolling_quantile_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    k = int(params["k"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    return np.asarray(t_hat, dtype=np.float64) * roll_med


def _rolling_quantile_ratio_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# frac_diff (R10c brainstorm #5c; fractional differencing, Lopez de Prado).
#
# T = (1 - L)^d y where L is the lag operator and ``d`` is a fractional order in (0, 1). The expansion truncates to ``lags`` terms with weight w_k = -w_{k-1} * (d - k + 1) / k starting at w_0 = 1. Inverse: y_hat = T_hat + (sum_{k=1}^{lags} w_k * y_{i-k}) using the SAME weight series.
#
# Preserves long-memory while making the target stationary -- proven win in finance / queue-length regimes. The inverse re-uses the past y values to reconstruct the level. Note: this is a UNARY-y transform (``base`` is unused but accepted for signature uniformity).
# ----------------------------------------------------------------------


def _frac_diff_weights(d: float, lags: int) -> np.ndarray:
    """Truncated weight series for (1 - L)^d expansion."""
    lags = max(1, int(lags))
    w = np.empty(lags + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, lags + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def _frac_diff_fit(
    y: np.ndarray, base: np.ndarray,
    d: float = _FRAC_DIFF_DEFAULT_D, lags: int = _FRAC_DIFF_DEFAULT_LAGS,
) -> Dict[str, Any]:
    """Store fractional order ``d``, lag truncation ``lags``, and the train-y mean used as a pre-window anchor (rows whose lag history is shorter than ``lags`` need a fallback value for the missing past terms)."""
    d = float(d)
    lags = max(1, int(lags))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f)
    anchor = float(np.mean(y_f[finite])) if finite.any() else 0.0
    return {"d": d, "lags": lags, "anchor": anchor, "weights": _frac_diff_weights(d, lags).tolist()}


def _frac_diff_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    """T_i = sum_{k=0}^{lags} w_k * y_{i-k}, padding y_{i-k} with the train anchor for k > i."""
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    anchor = float(params["anchor"])
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_f.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k_idx in range(min(i + 1, lags + 1)):
            s += weights[k_idx] * y_f[i - k_idx]
        # Pad missing past terms with the anchor (mean of train y).
        for k_idx in range(i + 1, lags + 1):
            s += weights[k_idx] * anchor
        out[i] = s
    return out


def _frac_diff_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    """Invert: T_i = w_0 * y_i + sum_{k=1}^{lags} w_k * y_{i-k}, so y_i = (T_i - sum_{k=1}^{lags} w_k * y_{i-k}) / w_0. w_0 == 1 by construction. Past y values are unknown at predict, so we ITERATIVELY reconstruct them: y_0 from T_0 + lag-anchors, y_1 from T_1 + y_0 + lag-anchors, etc."""
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    anchor = float(params["anchor"])
    t_f = np.asarray(t_hat, dtype=np.float64).reshape(-1)
    n = t_f.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lag_sum = 0.0
        for k_idx in range(1, min(i + 1, lags + 1)):
            lag_sum += weights[k_idx] * out[i - k_idx]
        for k_idx in range(i + 1, lags + 1):
            lag_sum += weights[k_idx] * anchor
        out[i] = (t_f[i] - lag_sum) / weights[0]
    return out


def _frac_diff_domain(
    y: Optional[np.ndarray], base: np.ndarray,
) -> np.ndarray:
    """Frac-diff is y-only but the contract accepts a base for signature uniformity. Domain: finite y; finite base (when provided)."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    base_ok = np.isfinite(base_f)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))












# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# Registry and lookup
# ----------------------------------------------------------------------

_TRANSFORMS_REGISTRY: Dict[str, Transform] = {
    "diff": Transform(
        name="diff",
        forward=_diff_forward,
        inverse=_diff_inverse,
        fit=_diff_fit,
        domain_check=_diff_domain,
        description="T = y - base. Inverse y_hat = T_hat + base. No fitted parameters.",
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "ratio": Transform(
        name="ratio",
        forward=_ratio_forward,
        inverse=_ratio_inverse,
        fit=_ratio_fit,
        domain_check=_ratio_domain,
        description=(
            "T = y / base. Inverse y_hat = T_hat * base. Requires |base| > 0; "
            "fitted eps stored from train scale."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "logratio": Transform(
        name="logratio",
        forward=_logratio_forward,
        inverse=_logratio_inverse,
        fit=_logratio_fit,
        domain_check=_logratio_domain,
        description=(
            "T = log(y) - log(base). Inverse y_hat = base * exp(softcap(T_hat)). "
            "Requires y, base > 0."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "linear_residual": Transform(
        name="linear_residual",
        forward=_linear_residual_forward,
        inverse=_linear_residual_inverse,
        fit=_linear_residual_fit,
        domain_check=_linear_residual_domain,
        description=(
            "T = y - alpha*base - beta with (alpha, beta) fitted via OLS on train. "
            "Inverse y_hat = T_hat + alpha*base + beta."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "linear_residual_multi": Transform(
        name="linear_residual_multi",
        forward=_linear_residual_multi_forward,
        inverse=_linear_residual_multi_inverse,
        fit=_linear_residual_multi_fit,
        domain_check=_linear_residual_multi_domain,
        description=(
            "T = y - sum_j(alpha_j * base_j) - beta with joint-OLS (alphas, beta) "
            "over a K-column base matrix. Inverse y_hat = T_hat + base @ alphas + beta. "
            "Falls back to zero-alpha + train mean intercept when the design-matrix "
            "condition number exceeds _MULTI_BASE_COND_NUMBER_MAX (multicollinearity guard)."
        ),
        tags=frozenset({TAG_CORE, TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "linear_residual_grouped": Transform(
        name="linear_residual_grouped",
        forward=_linear_residual_grouped_forward,
        inverse=_linear_residual_grouped_inverse,
        fit=_linear_residual_grouped_fit,
        domain_check=_linear_residual_grouped_domain,
        description=(
            "T = y - alpha_g * base - beta_g where g = group(row). Per-group OLS "
            "with James-Stein shrinkage toward global (alpha, beta). Small groups "
            "(n < _GROUPED_MIN_GROUP_SIZE) and unseen groups at predict time "
            "fall back to global. Requires a 'groups' kwarg threaded through "
            "fit/forward/inverse (wrapper extracts from configured group_column)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
    ),
    "quantile_residual": Transform(
        name="quantile_residual",
        forward=_quantile_residual_forward,
        inverse=_quantile_residual_inverse,
        fit=_quantile_residual_fit,
        domain_check=_quantile_residual_domain,
        description=(
            "Non-parametric heteroscedasticity-aware residual: T = (y - median_bin(y)) / IQR_bin(y) with ``n_bins`` quantile bins of ``base``. Inverse y_hat = T_hat * IQR_bin + median_bin. Under-populated bins (< min_bin_n train rows) and constant-y bins fall back to the global median(y) / IQR(y); out-of-range base values at predict map to the edge bin (no separate OOR bucket)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "monotonic_residual": Transform(
        name="monotonic_residual",
        forward=_monotonic_residual_forward,
        inverse=_monotonic_residual_inverse,
        fit=_monotonic_residual_fit,
        domain_check=_monotonic_residual_domain,
        description=(
            "T = y - g(base) where g is a monotone PCHIP spline fitted on quantile-knot medians. Generalises linear_residual to capture saturating / sigmoidal monotonic relationships that an OLS line leaves in the residual; PCHIP is monotone-preserving and the per-knot y-values are forced monotone (cumulative max/min along the Spearman-correlation orientation) so the interpolant is globally monotone. Out-of-range base values at predict clip to edge knot values (no PCHIP extrapolation)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "ewma_residual": Transform(
        name="ewma_residual",
        forward=_ewma_residual_forward,
        inverse=_ewma_residual_inverse,
        fit=_ewma_residual_fit,
        domain_check=_ewma_residual_domain,
        description=(
            "Time-ordered exponentially-weighted moving-average residual: T = y - EWMA_k(base) with alpha = 2/(k+1). Captures slow drift / regime persistence beyond a single lag. Caller is responsible for chronological row order at fit and predict; non-finite base values carry the previous EWMA state forward."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "rolling_quantile_ratio": Transform(
        name="rolling_quantile_ratio",
        forward=_rolling_quantile_ratio_forward,
        inverse=_rolling_quantile_ratio_inverse,
        fit=_rolling_quantile_ratio_fit,
        domain_check=_rolling_quantile_ratio_domain,
        description=(
            "Localised multiplicative residual: T = y / RollingMedian_k(base), with a centred window of ``k`` rows and an eps floor derived from train base scale to keep division safe at near-zero rolling medians. Inverse: y_hat = T_hat * RollingMedian_k(base). Like logratio but tracks the LOCAL base level instead of the global scale -- useful when y scales with a windowed median of base rather than the instantaneous value."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "frac_diff": Transform(
        name="frac_diff",
        forward=_frac_diff_forward,
        inverse=_frac_diff_inverse,
        fit=_frac_diff_fit,
        domain_check=_frac_diff_domain,
        description=(
            "Lopez de Prado fractional differencing: T_i = sum_k w_k * y_{i-k} with w_k = -w_{k-1} * (d - k + 1) / k truncated at ``lags`` terms. Preserves long-memory while making the target stationary. Inverse iteratively reconstructs y from T + the previously-reconstructed past terms. Pre-window padding uses the train-y mean."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
}


def get_transform(name: str) -> Transform:
    """Lookup helper. Raises :exc:`UnknownTransformError` for typos."""
    try:
        return _TRANSFORMS_REGISTRY[name]
    except KeyError:
        raise UnknownTransformError(
            f"Unknown transform '{name}'. Registered: {sorted(_TRANSFORMS_REGISTRY)}"
        )


# Short-name aliases for composite-target naming. Used in
# ``compose_target_name`` to keep displayed target names compact;
# previously composites were named ``TVT__linear_residual__TVT_prev``
# which read ugly in logs / report headings / dict keys. The dash
# separator + short aliases give us e.g. ``TVT-linres-TVT_prev``.
#
# Order: declared transforms only -- if a transform is missing from
# this map we fall back to the full name in ``compose_target_name`` so
# adding a new transform never silently breaks naming.
TRANSFORM_NAME_SHORT: Dict[str, str] = {
    "diff": "diff",
    "ratio": "ratio",
    "logratio": "logr",
    "linear_residual": "linres",
    "linear_residual_multi": "linresM",
    "linear_residual_grouped": "linresG",
    "quantile_residual": "qres",
    "monotonic_residual": "monres",
    "ewma_residual": "ewma",
    "rolling_quantile_ratio": "rqr",
    "frac_diff": "fdiff",
}


def compose_target_name(target_col: str, transform_name: str, base: str) -> str:
    """Build the canonical composite-target name from its three components.

    Uses ``-`` as the separator and the short transform alias from
    ``TRANSFORM_NAME_SHORT``. Falls back to the full transform name if
    the alias is missing (so brand-new transforms get a longer-but-correct
    name on day one instead of silent collision).

    Examples:
        compose_target_name('TVT', 'linear_residual', 'TVT_prev')
            -> 'TVT-linres-TVT_prev'
        compose_target_name('TVT', 'monotonic_residual', 'Y')
            -> 'TVT-monres-Y'
    """
    short = TRANSFORM_NAME_SHORT.get(transform_name, transform_name)
    return f"{target_col}-{short}-{base}"


# Reverse-lookup pattern fragments: ``f"-{short}-"`` and ``f"-{full}-"``
# both appear as substrings in canonical composite-target names. Used by
# ``is_composite_target_name`` to detect "this target name came from
# discovery, NOT a user-supplied column called ``y-base`` that happens
# to have one dash".
_COMPOSITE_NAME_FRAGMENTS: frozenset = frozenset(
    f"-{alias}-" for alias in TRANSFORM_NAME_SHORT.values()
) | frozenset(
    f"-{full}-" for full in TRANSFORM_NAME_SHORT.keys()
)


def is_composite_target_name(name: str) -> bool:
    """True if ``name`` matches the canonical composite-target naming
    convention ``{target}-{transform_short}-{base}`` for any registered
    transform.

    Used by per-target metric / chart helpers to switch their label from
    ``MTTR`` (raw mean target) to ``MTRESID`` (residual mean ~= 0 by
    construction). Robust to both the post-2026-05-13 short-alias format
    AND the legacy ``{target}__{transform}__{base}`` double-underscore
    format -- so loading a v1 suite-pickle still routes correctly.
    """
    if not name:
        return False
    if any(frag in name for frag in _COMPOSITE_NAME_FRAGMENTS):
        return True
    # Legacy double-underscore format (pre-2026-05-13 pickles).
    for full in TRANSFORM_NAME_SHORT.keys():
        if f"__{full}__" in name:
            return True
    return False


def list_transforms(*, tags: Optional[FrozenSet[str]] = None) -> List[str]:
    """Return registered transform names, optionally filtered by tag
    intersection (any-of: a transform passes if it has at least one of
    the requested tags)."""
    if tags is None:
        return sorted(_TRANSFORMS_REGISTRY)
    return sorted(
        name for name, t in _TRANSFORMS_REGISTRY.items() if t.tags & tags
    )
