"""``online_bayesian_linear_regression``, carved out of ``bayesian.py`` (X_EFFICIENCY_ARCHITECTURE-1
fix, mrmr_audit_2026-07-22) to clear the repo's enforced hard 1000-LOC CI gate (that file was 1001
lines). Behaviour preserved bit-for-bit; ``bayesian.py`` re-exports this function so every existing
``from mlframe.feature_engineering.bayesian import online_bayesian_linear_regression`` import keeps
working unchanged.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .bayesian import _NUMBA_AVAILABLE

if _NUMBA_AVAILABLE:
    from .bayesian import _oblr_groups_parallel, _oblr_groups_serial, _oblr_inner


def online_bayesian_linear_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    group_ids: Optional[np.ndarray] = None,
    prior_precision: float = 1.0,
    noise_sigma: float = 1.0,
    _force_backend: Optional[str] = None,
) -> dict:
    """Recursive Bayesian linear regression (NIG conjugate).

    Streaming online updates of the posterior over slope+intercept of
    ``y ~ X @ beta + noise``. Closed-form (no MCMC), O(k^2) per step.
    Returns dict with:
    * ``predictive_mean`` — one-step predictive mean ``X[t] @ E[beta]``
    * ``predictive_var`` — predictive variance (depends on row, not const)
    * ``log_marginal_lik`` — incremental log-evidence per row

    Use cases: finance beta tracking (slope drift over time), dose-
    response slope, ad-spend elasticity by day, sensor calibration
    coefficients. Distinct from KF on (k=1, X=ones): general case with
    multiple features and slope-uncertainty as a per-row feature.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.size:
        raise ValueError(f"X must be 2-D (n, k) with X.shape[0]=={y_arr.size}; got {X_arr.shape}")
    n, k = X_arr.shape
    out_pred_mean = np.full(n, np.nan, dtype=np.float64)
    out_pred_var = np.full(n, np.nan, dtype=np.float64)
    out_log_marg = np.full(n, np.nan, dtype=np.float64)

    def _run(idx_seg: np.ndarray) -> None:
        """Run the OBLR recursion over one segment, dispatching to the njit ``_oblr_inner`` kernel when available.

        Falls back to an equivalent pure-NumPy Sherman-Morrison-style covariance update when numba is unavailable.
        Used for the ``group_ids is None`` path and as the no-numba group fallback; the numba-group path routes
        through ``_oblr_groups_serial``/``_oblr_groups_parallel`` instead.
        """
        idx = np.ascontiguousarray(idx_seg, dtype=np.int64)
        if idx.size == 0:
            return
        if _NUMBA_AVAILABLE:
            y_sub = np.ascontiguousarray(y_arr[idx])
            X_sub = np.ascontiguousarray(X_arr[idx])
            pm = np.empty(idx.size, dtype=np.float64)
            pv = np.empty(idx.size, dtype=np.float64)
            lm = np.full(idx.size, np.nan, dtype=np.float64)
            _oblr_inner(y_sub, X_sub, prior_precision, noise_sigma, pm, pv, lm)
            out_pred_mean[idx] = pm
            out_pred_var[idx] = pv
            out_log_marg[idx] = lm
            return
        # Track covariance Sigma + Sherman-Morrison update: O(k^2) per step.
        mu = np.zeros(k, dtype=np.float64)
        Sigma = np.eye(k, dtype=np.float64) / prior_precision
        noise_var = noise_sigma**2
        for t in idx_seg:
            x_t = X_arr[t]
            y_t = float(y_arr[t])
            Sx = Sigma @ x_t
            pred_mean = float(x_t @ mu)
            pred_var = float(x_t @ Sx) + noise_var
            out_pred_mean[t] = pred_mean
            out_pred_var[t] = pred_var
            if np.isfinite(y_t):
                out_log_marg[t] = -0.5 * (math.log(2.0 * math.pi * pred_var) + (y_t - pred_mean) ** 2 / pred_var)
                innovation = y_t - pred_mean
                K = Sx / pred_var
                mu = mu + K * innovation
                Sigma = Sigma - np.outer(K, Sx)

    if group_ids is None:
        _run(np.arange(n))
    elif _NUMBA_AVAILABLE:
        from .grouped import iter_group_segments
        from ._recursion_dispatch import dispatch_recursion_backend
        sort_idx, starts, ends = iter_group_segments(group_ids)
        y_sorted = np.ascontiguousarray(y_arr[sort_idx])
        X_sorted = np.ascontiguousarray(X_arr[sort_idx])
        pm = np.empty(n, dtype=np.float64)
        pv = np.empty(n, dtype=np.float64)
        lm = np.full(n, np.nan, dtype=np.float64)
        backend = _force_backend if _force_backend in ("serial", "parallel") else dispatch_recursion_backend("fe_oblr", n, int(starts.size))
        if backend == "parallel":
            _oblr_groups_parallel(y_sorted, X_sorted, starts, ends, prior_precision, noise_sigma, pm, pv, lm)
        else:
            _oblr_groups_serial(y_sorted, X_sorted, starts, ends, prior_precision, noise_sigma, pm, pv, lm)
        out_pred_mean[sort_idx] = pm
        out_pred_var[sort_idx] = pv
        out_log_marg[sort_idx] = lm
    else:
        from .grouped import iter_group_segments
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _run(sort_idx[s:e])

    return {
        "predictive_mean": out_pred_mean,
        "predictive_var": out_pred_var,
        "log_marginal_lik": out_log_marg,
    }
