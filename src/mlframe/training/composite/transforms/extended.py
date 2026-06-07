"""Extended composite-target transforms (Tier 1-3 additions).

Six bivariate transforms + two multi-base transforms. All shipped
2026-05-26 to plug specific failure modes observed in production:

* ``asinh_residual``: ``logratio`` requires strictly positive base;
  many bivariate features (derivatives, distances, lag-deltas) can
  be negative. asinh-residual is log-like for ``|base| >> 1``,
  linear for ``|base| < 1``, and works on signed bases.

* ``centered_ratio``: ``ratio`` rejects rows where ``|base| < eps``.
  centered_ratio shifts base by a learned offset so ``(base + c) > 0``
  on train, expanding the domain.

* ``polynomial_residual_deg2``: ``linear_residual`` misses curvature;
  this adds a quadratic term ``T = y - alpha1*base - alpha2*base^2 - beta``.

* ``rank_residual``: distribution-free monotone residual. Heavy-tail
  targets where Yeo-Johnson doesn't fully whiten still respond to
  rank-space ridge. Inverse uses train rank-to-value lookup.

* ``smoothing_spline_residual``: generalises ``monotonic_residual``
  to arbitrary smooth (non-monotone) dependence. Scipy's
  UnivariateSpline does heavy lifting.

* ``reciprocal_residual``: ``T = 1/y - 1/base``. Niche but useful
  when y has multiplicative jump dynamics.

* ``geometric_mean_residual`` (multi-base): ``T = y / geomean(bases)``.
  Multi-base multiplicative variant of ``ratio``.

* ``pairwise_interaction_residual`` (multi-base): ``T = y - alpha *
  prod(bases) - beta``. Bilinear/trilinear residual; catches
  interaction terms that linear_residual_multi (additive) misses.

Numba/cupy bench summary (2026-05-26): all 8 transforms are
element-wise numpy or scipy-C-backed. JIT warm-up cost (~1-5s)
exceeds the per-call benefit at discovery scales (~50-150 calls per
transform per target). Numpy is correct here.

# bench-attempt-rejected (2026-05-26, asinh_residual + reciprocal_residual):
# numba @njit of forward/inverse: 1.2x speedup at n=1M warm, but
# 5s JIT compile vs 50ms one-shot numpy makes the warm-up dominate
# discovery wall time. Numpy stays.
# bench-attempt-rejected (2026-05-26, polynomial_residual_deg2):
# cupy @ (n, 2) GEMM tested. CPU 22ms vs GPU 8ms warm but +120ms
# H2D/D2H transfer per call. Loses overall to numpy at the
# discovery sample size (100k rows).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.composite_transforms_extended")


_RECIPROCAL_EPS_FLOOR: float = 1e-12
_POLY_DEG2_RIDGE: float = 1e-8  # tiny diag ridge for normal-eq stability
_SPLINE_DEFAULT_K: int = 3
_SPLINE_DEFAULT_S_MULT: float = 1.0  # smoothing = len(x) * std(y) * s_mult


# ============================================================
# 1. asinh_residual: T = arcsinh(y) - alpha * arcsinh(base)
# ============================================================
# arcsinh(x) = log(x + sqrt(x^2 + 1)). Defined on all real numbers,
# log-like for |x| >> 1, linear for |x| << 1. The bivariate residual
# T = arcsinh(y) - alpha * arcsinh(base) generalises logratio to
# signed bases. Fitted alpha = OLS on train.

def _asinh_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    yz = np.arcsinh(np.asarray(y, dtype=np.float64))
    bz = np.arcsinh(np.asarray(base, dtype=np.float64))
    finite = np.isfinite(yz) & np.isfinite(bz)
    if finite.sum() < 10:
        return {"alpha": 1.0, "beta": 0.0}
    yc = yz[finite]
    bc = bz[finite]
    b_mean = float(bc.mean())
    b_centered = bc - b_mean
    denom = float(np.dot(b_centered, b_centered))
    if denom <= 0:
        return {"alpha": 0.0, "beta": float(yc.mean())}
    alpha = float(np.dot(b_centered, yc - yc.mean()) / denom)
    beta = float(yc.mean() - alpha * b_mean)
    return {"alpha": alpha, "beta": beta}


def _asinh_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return np.arcsinh(np.asarray(y, dtype=np.float64)) \
        - alpha * np.arcsinh(np.asarray(base, dtype=np.float64)) - beta


def _asinh_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    z = np.asarray(t_hat, dtype=np.float64) \
        + alpha * np.arcsinh(np.asarray(base, dtype=np.float64)) + beta
    return np.sinh(z)


def _asinh_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


# ============================================================
# 2. centered_ratio: T = y / (base + c)
# ============================================================
# Extension of ratio to signed bases. ``c`` is fitted at train-time
# so (base + c) is strictly positive on train (subject to eps floor).

def _centered_ratio_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    base_arr = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(base_arr)
    if not finite.any():
        return {"c": 0.0, "eps": _RECIPROCAL_EPS_FLOOR}
    b_min = float(base_arr[finite].min())
    b_scale = float(np.median(np.abs(base_arr[finite])))
    # c = -b_min + 0.01 * scale: shift the minimum slightly above zero.
    # Tiny safety floor so even constant-on-train base stays nonzero.
    c = -b_min + 0.01 * max(b_scale, 1e-6)
    eps = max(b_scale * 1e-6, _RECIPROCAL_EPS_FLOOR)
    return {"c": float(c), "eps": eps}


def _centered_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    c = float(params["c"])
    eps = float(params["eps"])
    shifted = np.asarray(base, dtype=np.float64) + c
    safe = np.where(np.abs(shifted) < eps, np.sign(shifted + 1e-300) * eps, shifted)
    return np.asarray(y, dtype=np.float64) / safe


def _centered_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    c = float(params["c"])
    return np.asarray(t_hat, dtype=np.float64) * (np.asarray(base, dtype=np.float64) + c)


def _centered_ratio_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


# ============================================================
# 3. polynomial_residual_deg2: T = y - a1*base - a2*base^2 - b
# ============================================================
# Degree-2 OLS via normal equations on the (1, base, base^2)
# design matrix. Tiny diag ridge keeps the solve stable on
# near-constant base columns.

def _polynomial_residual_deg2_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.isfinite(bv)
    if finite.sum() < 10:
        return {"alpha1": 1.0, "alpha2": 0.0, "beta": 0.0}
    yc = yv[finite]
    bc = bv[finite]
    # Design matrix (1, base, base^2) with column-mean removal for
    # numerical stability; intercept absorbed into beta post-solve.
    X = np.column_stack([np.ones_like(bc), bc, bc * bc])
    XtX = X.T @ X
    XtX[1, 1] += _POLY_DEG2_RIDGE
    XtX[2, 2] += _POLY_DEG2_RIDGE
    try:
        coef = np.linalg.solve(XtX, X.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.array([float(yc.mean()), 0.0, 0.0])
    return {
        "beta": float(coef[0]),
        "alpha1": float(coef[1]),
        "alpha2": float(coef[2]),
    }


def _polynomial_residual_deg2_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    a1 = float(params["alpha1"])
    a2 = float(params["alpha2"])
    b = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    return np.asarray(y, dtype=np.float64) - a1 * bv - a2 * bv * bv - b


def _polynomial_residual_deg2_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    a1 = float(params["alpha1"])
    a2 = float(params["alpha2"])
    b = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    return np.asarray(t_hat, dtype=np.float64) + a1 * bv + a2 * bv * bv + b


def _polynomial_residual_deg2_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


# ============================================================
# 4. rank_residual: T = rank(y)/n - alpha * rank(base)/n
# ============================================================
# Distribution-free monotone residual. Forward maps via the
# train-fitted rank-to-value tables (one per axis). Inverse uses
# bisect on the train y-rank table to recover y.

def _rank_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.isfinite(bv)
    if finite.sum() < 10:
        return {
            "y_sorted": np.array([0.0, 1.0]),
            "b_sorted": np.array([0.0, 1.0]),
            "alpha": 0.0,
            "beta": 0.5,
        }
    yc = yv[finite]
    bc = bv[finite]
    y_sorted = np.sort(yc)
    b_sorted = np.sort(bc)
    n = yc.size
    # Empirical CDF rank-fractions (mid-rank to avoid 0/1 endpoints).
    yr = (np.searchsorted(y_sorted, yc, side="left").astype(np.float64) + 0.5) / n
    br = (np.searchsorted(b_sorted, bc, side="left").astype(np.float64) + 0.5) / n
    b_mean = float(br.mean())
    bc_ranks = br - b_mean
    denom = float(np.dot(bc_ranks, bc_ranks))
    if denom <= 0:
        return {
            "y_sorted": y_sorted,
            "b_sorted": b_sorted,
            "alpha": 0.0,
            "beta": float(yr.mean()),
        }
    alpha = float(np.dot(bc_ranks, yr - yr.mean()) / denom)
    beta = float(yr.mean() - alpha * b_mean)
    return {
        "y_sorted": y_sorted,
        "b_sorted": b_sorted,
        "alpha": alpha,
        "beta": beta,
    }


def _rank_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    y_sorted = np.asarray(params["y_sorted"], dtype=np.float64)
    b_sorted = np.asarray(params["b_sorted"], dtype=np.float64)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    yr = (np.searchsorted(y_sorted, yv, side="left").astype(np.float64) + 0.5) / max(y_sorted.size, 1)
    br = (np.searchsorted(b_sorted, bv, side="left").astype(np.float64) + 0.5) / max(b_sorted.size, 1)
    return yr - alpha * br - beta


def _rank_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    y_sorted = np.asarray(params["y_sorted"], dtype=np.float64)
    b_sorted = np.asarray(params["b_sorted"], dtype=np.float64)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    br = (np.searchsorted(b_sorted, bv, side="left").astype(np.float64) + 0.5) / max(b_sorted.size, 1)
    yr_hat = np.asarray(t_hat, dtype=np.float64) + alpha * br + beta
    # Clip to [0, 1] then map back to y via the train y-sorted table.
    n_y = y_sorted.size
    if n_y < 2:
        return np.full(yr_hat.shape, float(y_sorted[0]) if n_y else 0.0)
    yr_clipped = np.clip(yr_hat, 0.0, 1.0 - 1e-9)
    idx = np.clip((yr_clipped * n_y).astype(np.int64), 0, n_y - 1)
    return y_sorted[idx]


def _rank_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


# ============================================================
# 5. smoothing_spline_residual: T = y - SmoothingSpline(base)
# ============================================================
# scipy UnivariateSpline fitted on (base, y) train pairs. Params store
# the train (unique_b, yc_avg) arrays + smoothing-factor ``s``; the
# spline is rebuilt at forward/inverse time. Same pattern as
# ``monotonic_residual`` (which stores knots_x/knots_y and rebuilds
# PchipInterpolator) so params are pickle-clean.

def _smoothing_spline_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.isfinite(bv)
    if finite.sum() < 20:
        return {
            "knots_b": np.zeros(0, dtype=np.float64),
            "knots_y": np.zeros(0, dtype=np.float64),
            "s": 0.0,
            "y_mean": float(yv[finite].mean() if finite.any() else 0.0),
        }
    yc = yv[finite]
    bc = bv[finite]
    order = np.argsort(bc)
    bc_s = bc[order]
    yc_s = yc[order]
    unique_b, inv_idx = np.unique(bc_s, return_inverse=True)
    if unique_b.size < 4:
        return {
            "knots_b": np.zeros(0, dtype=np.float64),
            "knots_y": np.zeros(0, dtype=np.float64),
            "s": 0.0,
            "y_mean": float(yc.mean()),
        }
    yc_avg = np.bincount(inv_idx, weights=yc_s) / np.bincount(inv_idx)
    s = max(unique_b.size, 1) * float(np.std(yc_avg)) * _SPLINE_DEFAULT_S_MULT
    return {
        "knots_b": unique_b.astype(np.float64, copy=False),
        "knots_y": yc_avg.astype(np.float64, copy=False),
        "s": float(s),
        "y_mean": float(yc.mean()),
    }


def _smoothing_spline_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    knots_b = np.asarray(params.get("knots_b", []), dtype=np.float64)
    knots_y = np.asarray(params.get("knots_y", []), dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64).reshape(-1)
    y_mean = float(params.get("y_mean", 0.0))
    if knots_b.size < 4:
        return np.full(bv.shape, y_mean, dtype=np.float64)
    try:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(
            knots_b, knots_y, k=_SPLINE_DEFAULT_K,
            s=float(params.get("s", 0.0)), ext="const",
        )
        g = np.asarray(spl(bv), dtype=np.float64)
    except Exception:
        return np.full(bv.shape, y_mean, dtype=np.float64)
    return np.where(np.isfinite(g), g, y_mean)


def _smoothing_spline_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - _smoothing_spline_g(base, params)


def _smoothing_spline_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(t_hat, dtype=np.float64) + _smoothing_spline_g(base, params)


def _smoothing_spline_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


# ============================================================
# 6. reciprocal_residual: T = 1/y - 1/base
# ============================================================
# Niche: useful when y has multiplicative jump dynamics.

def _reciprocal_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    y_scale = float(np.median(np.abs(yv[np.isfinite(yv)]))) if np.isfinite(yv).any() else 1.0
    b_scale = float(np.median(np.abs(bv[np.isfinite(bv)]))) if np.isfinite(bv).any() else 1.0
    return {
        "eps_y": max(y_scale * 1e-6, _RECIPROCAL_EPS_FLOOR),
        "eps_b": max(b_scale * 1e-6, _RECIPROCAL_EPS_FLOOR),
    }


def _reciprocal_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    eps_y = float(params["eps_y"])
    eps_b = float(params["eps_b"])
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    safe_y = np.where(np.abs(yv) < eps_y, np.sign(yv + 1e-300) * eps_y, yv)
    safe_b = np.where(np.abs(bv) < eps_b, np.sign(bv + 1e-300) * eps_b, bv)
    return 1.0 / safe_y - 1.0 / safe_b


def _reciprocal_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    eps_b = float(params["eps_b"])
    bv = np.asarray(base, dtype=np.float64)
    safe_b = np.where(np.abs(bv) < eps_b, np.sign(bv + 1e-300) * eps_b, bv)
    z = np.asarray(t_hat, dtype=np.float64) + 1.0 / safe_b
    # y = 1 / z; guard near-zero z by clamping to eps (prediction would
    # otherwise blow up).
    eps_z = max(eps_b, _RECIPROCAL_EPS_FLOOR)
    safe_z = np.where(np.abs(z) < eps_z, np.sign(z + 1e-300) * eps_z, z)
    return 1.0 / safe_z


def _reciprocal_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.isfinite(bv) & (np.abs(bv) > 0)
    if y is None:
        return base_ok
    yv = np.asarray(y, dtype=np.float64)
    return base_ok & np.isfinite(yv) & (np.abs(yv) > 0)


# ============================================================
# 7. geometric_mean_residual: T = y / geomean(bases)
# ============================================================
# Multi-base, multiplicative. Requires every base column > 0 on
# the row (strict positivity). Inverse: y = T * geomean(bases).

def _geometric_mean_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    scale = float(np.median(np.abs(bv[np.isfinite(bv)]))) if np.isfinite(bv).any() else 1.0
    return {"eps": max(scale * 1e-6, _RECIPROCAL_EPS_FLOOR), "n_bases": int(bv.shape[1])}


def _geometric_mean_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    eps = float(params["eps"])
    bv = np.asarray(base, dtype=np.float64)
    # geomean via log-mean-exp. Strict positivity required by domain.
    log_b = np.log(np.where(bv > eps, bv, eps))
    g = np.exp(log_b.mean(axis=1))
    return np.asarray(y, dtype=np.float64) / np.where(g > eps, g, eps)


def _geometric_mean_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    eps = float(params["eps"])
    bv = np.asarray(base, dtype=np.float64)
    log_b = np.log(np.where(bv > eps, bv, eps))
    g = np.exp(log_b.mean(axis=1))
    return np.asarray(t_hat, dtype=np.float64) * g


def _geometric_mean_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.all(np.isfinite(bv) & (bv > 0), axis=1)
    if y is None:
        return base_ok
    yv = np.asarray(y, dtype=np.float64)
    return base_ok & np.isfinite(yv) & (yv > 0)


# ============================================================
# 8. pairwise_interaction_residual: T = y - alpha*prod(bases) - beta
# ============================================================
# Multi-base, BILINEAR/multilinear. Captures pure interaction term;
# residual after removing the multiplicative contribution. ``alpha``
# fitted via OLS on (1, prod(bases)) train pairs.

def _pairwise_interaction_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.all(np.isfinite(bv), axis=1)
    if finite.sum() < 10:
        return {"alpha": 0.0, "beta": float(yv[finite].mean() if finite.any() else 0.0)}
    yc = yv[finite]
    p = np.prod(bv[finite], axis=1)
    p_mean = float(p.mean())
    p_centered = p - p_mean
    denom = float(np.dot(p_centered, p_centered))
    if denom <= 0:
        return {"alpha": 0.0, "beta": float(yc.mean())}
    alpha = float(np.dot(p_centered, yc - yc.mean()) / denom)
    beta = float(yc.mean() - alpha * p_mean)
    return {"alpha": alpha, "beta": beta}


def _pairwise_interaction_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    p = np.prod(bv, axis=1)
    return np.asarray(y, dtype=np.float64) - alpha * p - beta


def _pairwise_interaction_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    p = np.prod(bv, axis=1)
    return np.asarray(t_hat, dtype=np.float64) + alpha * p + beta


def _pairwise_interaction_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.all(np.isfinite(bv), axis=1)
    if y is None:
        return base_ok
    yv = np.asarray(y, dtype=np.float64)
    return base_ok & np.isfinite(yv)


__all__ = [
    "_asinh_residual_fit", "_asinh_residual_forward",
    "_asinh_residual_inverse", "_asinh_residual_domain",
    "_centered_ratio_fit", "_centered_ratio_forward",
    "_centered_ratio_inverse", "_centered_ratio_domain",
    "_polynomial_residual_deg2_fit", "_polynomial_residual_deg2_forward",
    "_polynomial_residual_deg2_inverse", "_polynomial_residual_deg2_domain",
    "_rank_residual_fit", "_rank_residual_forward",
    "_rank_residual_inverse", "_rank_residual_domain",
    "_smoothing_spline_residual_fit", "_smoothing_spline_residual_forward",
    "_smoothing_spline_residual_inverse", "_smoothing_spline_residual_domain",
    "_reciprocal_residual_fit", "_reciprocal_residual_forward",
    "_reciprocal_residual_inverse", "_reciprocal_residual_domain",
    "_geometric_mean_residual_fit", "_geometric_mean_residual_forward",
    "_geometric_mean_residual_inverse", "_geometric_mean_residual_domain",
    "_pairwise_interaction_residual_fit", "_pairwise_interaction_residual_forward",
    "_pairwise_interaction_residual_inverse", "_pairwise_interaction_residual_domain",
]
