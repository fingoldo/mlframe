"""Unary y-only composite transforms + chain meta-transform.

The bivariate registry transforms each take ``(y, base) -> T`` where ``base`` is a dominant feature column (e.g. a lag feature). For heavy-tailed targets where the dominant feature has been already absorbed by a bivariate composite (e.g. ``y-linres-lag1``) the remaining residual T is still skewed / leptokurtic; a unary y-only transform on T (or on raw y) can compress the tails further.

Production motivator: composite-CB on ``y-linres-lag1`` reported ``excess_kurt=+2.40`` on the residual; the recommended inner loss is MAE, but a unary ``cbrt`` / ``yeo_johnson`` applied to the residual would compress tails further and let RMSE-trained inners produce stable predictions.

This module ships:

- 4 unary y-transforms as pure-numpy ``forward / inverse / fit / domain_check`` tuples, no base argument:

  - ``cbrt_y`` -- signed cube-root: ``T = sign(y) * |y|^(1/3)``. Inverse: ``y = T^3``. No fitted params. Defined for all real y. Compresses heavy tails without breaking sign.
  - ``log_y`` -- shifted log: ``T = log(y + offset)`` where offset is fitted so ``min(y_train) + offset > 0``. Inverse: ``y = exp(T) - offset``. Caller restricted to ``y > -offset``.
  - ``yeo_johnson_y`` -- Yeo-Johnson power transform: ``lambda`` fitted via MLE on train. Works on all real y (unlike Box-Cox). Inverse uses the closed-form Yeo-Johnson inverse.
  - ``quantile_normal_y`` -- empirical-CDF -> standard Normal: ``T = Phi^{-1}(rank(y) / (n+1))``. Inverse interpolates the fitted CDF. Robust to any monotone distortion of y but loses absolute scale.

- A ``chain_bivariate_then_unary`` composer that:
  - applies a bivariate transform first (``T1 = bivariate.forward(y, base)``),
  - then applies a unary transform on T1 (``T2 = unary.forward(T1)``),
  - inverts by reverse order at predict.

These helpers are pure numpy; integration into the registry + ``CompositeTargetEstimator`` wrapper (adding a ``requires_base: bool = True`` field to the Transform dataclass and skipping base-column extraction when ``False``) is the follow-up step. Tests pin the round-trip math at the helper level so the wrapper integration cannot regress silently.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

try:
    import numba as _numba
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep but allow graceful skip
    _numba = None  # type: ignore
    _HAS_NUMBA = False

# numba JIT'd YJ forward/inverse beats numpy fancy-indexing from n>=~10k
# upward (bench at tests/perf/bench_yj_forward.py). Below this threshold the
# numpy path's lower per-call overhead wins, and the JIT compile cost
# (one-shot, but still ~100ms) doesn't amortise over a single Brent step on
# tiny data.
_YJ_NUMBA_MIN_N: int = 10_000

# Yeo-Johnson inverse has a lambda-dependent asymptote: for lam<0 the nonneg branch base ``t*lam+1`` reaches 0 at t=-1/lam (y->+inf) and goes negative beyond it; for lam>2 the neg branch base does the same. ``base**(1/lam)`` on a non-positive base is NaN, which ``np.clip`` cannot repair, so an out-of-range inner T-prediction silently poisoned predict()/predict_pre_clip. We floor the base to a tiny positive value so the inverse SATURATES at the asymptote (large finite y, then bounded by the post-inverse y-clip) instead of returning NaN. Floor only bites within ~1e-12 of the asymptote, so the forward/inverse round-trip stays value-identical on every realistic y.
_YJ_INV_BASE_FLOOR: float = 1e-12


if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=True, parallel=True)
    def _yj_forward_numba_kernel(y, lam):  # type: ignore[no-untyped-def]
        n = y.shape[0]
        out = np.empty(n, dtype=np.float64)
        lam_is_zero = abs(lam) < 1e-12
        lam_is_two = abs(lam - 2.0) < 1e-12
        inv_lam = 0.0 if lam_is_zero else 1.0 / lam
        two_minus_lam = 2.0 - lam
        inv_2ml = 0.0 if lam_is_two else 1.0 / two_minus_lam
        for i in _numba.prange(n):
            yi = y[i]
            if yi >= 0.0:
                if lam_is_zero:
                    out[i] = np.log1p(yi)
                else:
                    out[i] = ((yi + 1.0) ** lam - 1.0) * inv_lam
            else:
                if lam_is_two:
                    out[i] = -np.log1p(-yi)
                else:
                    out[i] = -((-yi + 1.0) ** two_minus_lam - 1.0) * inv_2ml
        return out

    # fastmath=False on the inverse kernel: predict-time correctness needs
    # bit-exact match vs the numpy reference (the Brent fit hotspot is on
    # the forward path only, so dropping fastmath here costs nothing the
    # 5-10x speedup target cares about).
    @_numba.njit(cache=True, fastmath=False, parallel=True)
    def _yj_inverse_numba_kernel(t, lam):  # type: ignore[no-untyped-def]
        n = t.shape[0]
        out = np.empty(n, dtype=np.float64)
        lam_is_zero = abs(lam) < 1e-12
        lam_is_two = abs(lam - 2.0) < 1e-12
        inv_lam = 0.0 if lam_is_zero else 1.0 / lam
        two_minus_lam = 2.0 - lam
        inv_2ml = 0.0 if lam_is_two else 1.0 / two_minus_lam
        for i in _numba.prange(n):
            ti = t[i]
            if ti >= 0.0:
                if lam_is_zero:
                    out[i] = np.expm1(ti)
                else:
                    base = ti * lam + 1.0
                    if base < _YJ_INV_BASE_FLOOR:
                        base = _YJ_INV_BASE_FLOOR
                    out[i] = base ** inv_lam - 1.0
            else:
                if lam_is_two:
                    out[i] = -np.expm1(-ti)
                else:
                    base = -ti * two_minus_lam + 1.0
                    if base < _YJ_INV_BASE_FLOOR:
                        base = _YJ_INV_BASE_FLOOR
                    out[i] = -(base ** inv_2ml - 1.0)
        return out


# ----------------------------------------------------------------------
# cbrt_y -- signed cube-root, all real y, no fitted params.
# ----------------------------------------------------------------------

def cbrt_y_fit(y: np.ndarray) -> Dict[str, Any]:
    return {}


def cbrt_y_forward(y: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    return np.cbrt(np.asarray(y, dtype=np.float64))


def cbrt_y_inverse(t: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    arr = np.asarray(t, dtype=np.float64)
    return arr * arr * arr


def cbrt_y_domain(y: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(y, dtype=np.float64))


# ----------------------------------------------------------------------
# log_y -- shifted log; ``offset`` fitted so min(y_train) + offset > 0.
# ----------------------------------------------------------------------

_LOG_OFFSET_SAFETY: float = 1.0
"""Additional positive margin above |min(y_train)| so log doesn't see exactly zero (eps-floor for floating-point safety)."""


def log_y_fit(y: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(y, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"offset": _LOG_OFFSET_SAFETY}
    y_min = float(finite.min())
    offset = -y_min + _LOG_OFFSET_SAFETY if y_min <= 0.0 else _LOG_OFFSET_SAFETY
    return {"offset": offset}


def log_y_forward(y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    offset = float(params["offset"])
    return np.log(np.asarray(y, dtype=np.float64) + offset)


def log_y_inverse(t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    offset = float(params["offset"])
    return np.exp(np.asarray(t, dtype=np.float64)) - offset


def log_y_domain(y: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if params is None:
        return np.isfinite(arr)
    offset = float(params["offset"])
    return np.isfinite(arr) & (arr + offset > 0.0)


# ----------------------------------------------------------------------
# yeo_johnson_y -- Yeo-Johnson power transform with fitted lambda.
# ----------------------------------------------------------------------
# The Yeo-Johnson transform is a smooth generalisation of Box-Cox that supports negative y. The closed-form definition is:
#   y >= 0, lambda != 0: ((y + 1)^lambda - 1) / lambda
#   y >= 0, lambda == 0: log(y + 1)
#   y <  0, lambda != 2: -(((-y + 1)^(2 - lambda) - 1) / (2 - lambda))
#   y <  0, lambda == 2: -log(-y + 1)
# Inverse mirrors with branch on sign(t) and matching lambda branch.


def _yj_forward_scalar(y: float, lam: float) -> float:
    if y >= 0.0:
        if abs(lam) < 1e-12:
            return float(np.log1p(y))
        return float((np.power(y + 1.0, lam) - 1.0) / lam)
    if abs(lam - 2.0) < 1e-12:
        return float(-np.log1p(-y))
    return float(-(np.power(-y + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam))


def _yj_inverse_scalar(t: float, lam: float) -> float:
    if t >= 0.0:
        if abs(lam) < 1e-12:
            return float(np.expm1(t))
        base = max(t * lam + 1.0, _YJ_INV_BASE_FLOOR)
        return float(np.power(base, 1.0 / lam) - 1.0)
    if abs(lam - 2.0) < 1e-12:
        return float(-np.expm1(-t))
    base = max(-t * (2.0 - lam) + 1.0, _YJ_INV_BASE_FLOOR)
    return float(-(np.power(base, 1.0 / (2.0 - lam)) - 1.0))


def _yj_forward_numpy(y: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    nonneg = y >= 0.0
    pos = y[nonneg]
    neg = y[~nonneg]
    if abs(lam) < 1e-12:
        out[nonneg] = np.log1p(pos)
    else:
        out[nonneg] = (np.power(pos + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-12:
        out[~nonneg] = -np.log1p(-neg)
    else:
        out[~nonneg] = -(np.power(-neg + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _yj_inverse_numpy(t: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(t, dtype=np.float64)
    nonneg = t >= 0.0
    pos = t[nonneg]
    neg = t[~nonneg]
    if abs(lam) < 1e-12:
        out[nonneg] = np.expm1(pos)
    else:
        base_pos = np.maximum(pos * lam + 1.0, _YJ_INV_BASE_FLOOR)
        out[nonneg] = np.power(base_pos, 1.0 / lam) - 1.0
    if abs(lam - 2.0) < 1e-12:
        out[~nonneg] = -np.expm1(-neg)
    else:
        base_neg = np.maximum(-neg * (2.0 - lam) + 1.0, _YJ_INV_BASE_FLOOR)
        out[~nonneg] = -(np.power(base_neg, 1.0 / (2.0 - lam)) - 1.0)
    return out


def _yj_forward(y: np.ndarray, lam: float) -> np.ndarray:
    """Yeo-Johnson forward transform. Size-dispatches to a numba parallel
    kernel when ``n >= _YJ_NUMBA_MIN_N`` (5-10x speedup at n>=50k, bench
    at tests/perf/bench_yj_forward.py). Falls back to the numpy reference
    on tiny inputs or when numba is unavailable; the numba path matches
    the numpy reference within ~1e-13 (fastmath=True)."""
    if _HAS_NUMBA and y.shape[0] >= _YJ_NUMBA_MIN_N:
        return _yj_forward_numba_kernel(y, lam)
    return _yj_forward_numpy(y, lam)


def _yj_inverse(t: np.ndarray, lam: float) -> np.ndarray:
    """Yeo-Johnson inverse transform. See ``_yj_forward`` for the
    dispatch contract; same numba/numpy size-dispatch."""
    if _HAS_NUMBA and t.shape[0] >= _YJ_NUMBA_MIN_N:
        return _yj_inverse_numba_kernel(t, lam)
    return _yj_inverse_numpy(t, lam)


def yeo_johnson_y_fit(y: np.ndarray) -> Dict[str, Any]:
    """Fit lambda via Brent minimisation of negative log-likelihood (MLE) on a normality-fit objective. Falls back to lambda=1.0 (identity-ish) on numerical failure.

    Implementation note: scipy.stats.yeojohnson_normmax / sklearn PowerTransformer would both work but pull heavy deps; the closed-form likelihood here keeps the file self-contained for the unit tests. The objective is the standard YJ profile log-likelihood (constant variance + Jacobian correction). Range for lambda is clipped to [-2, 4] which covers all practical heavy-tail compression / expansion needs.
    """
    arr = np.asarray(y, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 4:
        return {"lambda": 1.0}
    n = float(finite.size)

    def neg_loglik(lam: float) -> float:
        t = _yj_forward(finite, lam)
        var = float(t.var())
        if var <= 0.0:
            return float("inf")
        # YJ log-Jacobian: sum sign(y) * (lambda - 1) * log(|y| + 1)... but we use the equivalent form: profile likelihood with Jacobian correction subsumed into the variance term and a (lambda - 1) sum log term.
        log_jac = float(np.sum(np.sign(finite) * (lam - 1.0) * np.log(np.abs(finite) + 1.0)))
        return 0.5 * n * np.log(var) - log_jac

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(neg_loglik, bounds=(-2.0, 4.0), method="bounded", options={"xatol": 1e-4})
        lam = float(result.x) if result.success else 1.0
    except Exception:
        lam = 1.0
    return {"lambda": float(np.clip(lam, -2.0, 4.0))}


def yeo_johnson_y_forward(y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return _yj_forward(np.asarray(y, dtype=np.float64), float(params["lambda"]))


def yeo_johnson_y_inverse(t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return _yj_inverse(np.asarray(t, dtype=np.float64), float(params["lambda"]))


def yeo_johnson_y_domain(y: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    return np.isfinite(np.asarray(y, dtype=np.float64))


# ----------------------------------------------------------------------
# quantile_normal_y -- empirical-CDF mapping y -> standard Normal.
# ----------------------------------------------------------------------

def quantile_normal_y_fit(y: np.ndarray, n_quantiles: int = 1000) -> Dict[str, Any]:
    """Fit empirical CDF as a knot table. At predict-time the wrapper interpolates the y -> uniform map then applies the standard-Normal quantile function.

    ``n_quantiles`` controls the knot density; 1000 captures the empirical distribution to <0.1% percentile error on practical N.
    """
    arr = np.asarray(y, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return {"knots_y": np.array([0.0, 1.0]), "knots_q": np.array([0.0, 1.0])}
    sorted_y = np.sort(finite)
    n = sorted_y.size
    # Equi-spaced ranks 1..n_quantiles taken from the sorted sample.
    n_knots = int(min(n_quantiles, n))
    idx = np.linspace(0, n - 1, n_knots).astype(np.int64)
    knots_y = sorted_y[idx]
    # Smith's plotting position (i - 0.5) / n is unbiased for the median rank of order statistics under continuous F; matches sklearn QuantileTransformer convention.
    knots_q = (idx + 0.5) / n
    return {"knots_y": knots_y, "knots_q": knots_q}


def quantile_normal_y_forward(y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    from scipy.stats import norm
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    knots_q = np.asarray(params["knots_q"], dtype=np.float64)
    arr = np.asarray(y, dtype=np.float64)
    # Map y -> uniform via interp; clip to (eps, 1-eps) so the Normal quantile stays finite at the tails.
    q = np.interp(arr, knots_y, knots_q)
    eps = 1.0 / (2.0 * len(knots_q))
    q = np.clip(q, eps, 1.0 - eps)
    return norm.ppf(q)


def quantile_normal_y_inverse(t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    from scipy.stats import norm
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    knots_q = np.asarray(params["knots_q"], dtype=np.float64)
    q = norm.cdf(np.asarray(t, dtype=np.float64))
    # Inverse interp: knots_q -> knots_y.
    return np.interp(q, knots_q, knots_y)


def quantile_normal_y_domain(y: np.ndarray, params: Dict[str, Any] | None = None) -> np.ndarray:
    return np.isfinite(np.asarray(y, dtype=np.float64))


# ----------------------------------------------------------------------
# chain_bivariate_then_unary -- compose a (y, base) -> T1 bivariate with a T1 -> T2 unary.
# ----------------------------------------------------------------------
# Take a SURVIVING bivariate composite spec (e.g. ``linear_residual`` on a lag base) and stack one of the unary transforms above on top. The model learns the doubly-transformed target; at predict time the inverse runs unary first, then bivariate.

UnaryFns = Tuple[
    Callable[[np.ndarray], Dict[str, Any]],
    Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
]
"""(fit, forward, inverse) for the unary half of a chain."""


def chain_bivariate_then_unary_fit(
    y: np.ndarray,
    base: np.ndarray,
    bivariate_fit: Callable[[np.ndarray, np.ndarray], Dict[str, Any]],
    bivariate_forward: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary: UnaryFns,
) -> Dict[str, Any]:
    """Fit the bivariate on (y, base), apply forward to get T1, then fit the unary on T1."""
    bi_params = bivariate_fit(y, base)
    t1 = bivariate_forward(y, base, bi_params)
    unary_fit_fn = unary[0]
    un_params = unary_fit_fn(t1)
    return {"bivariate_params": bi_params, "unary_params": un_params}


def chain_bivariate_then_unary_forward(
    y: np.ndarray,
    base: np.ndarray,
    params: Dict[str, Any],
    bivariate_forward: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary: UnaryFns,
) -> np.ndarray:
    t1 = bivariate_forward(y, base, params["bivariate_params"])
    return unary[1](t1, params["unary_params"])


def chain_bivariate_then_unary_inverse(
    t2: np.ndarray,
    base: np.ndarray,
    params: Dict[str, Any],
    bivariate_inverse: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary: UnaryFns,
) -> np.ndarray:
    t1_hat = unary[2](t2, params["unary_params"])
    return bivariate_inverse(t1_hat, base, params["bivariate_params"])


# ----------------------------------------------------------------------
# Multi-stage chains: bivariate + 2+ unary stages. E.g.
# chain([linres, cbrt, quantile_normal]) for very heavy-tail residuals
# where a single unary does not bring the distribution close to Gaussian.
# The composer applies stages in order at forward time and in reverse order
# at inverse time.
# ----------------------------------------------------------------------


def chain_multi_stage_fit(
    y: np.ndarray,
    base: np.ndarray,
    *,
    bivariate_fit: Callable[[np.ndarray, np.ndarray], Dict[str, Any]],
    bivariate_forward: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary_stages: list[UnaryFns],
) -> Dict[str, Any]:
    """Fit a bivariate composite + N unary stages sequentially.

    Returns ``{"bivariate_params": ..., "unary_stage_params": [params_1, params_2, ...]}``.
    Each unary's ``fit`` runs on the OUTPUT of the previous stage so the per-stage params reflect the actual distribution that stage sees.
    """
    bi_params = bivariate_fit(y, base)
    t = bivariate_forward(y, base, bi_params)
    stage_params: list[Dict[str, Any]] = []
    for fit_fn, forward_fn, _inv_fn in unary_stages:
        p = fit_fn(t)
        stage_params.append(p)
        t = forward_fn(t, p)
    return {"bivariate_params": bi_params, "unary_stage_params": stage_params}


def chain_multi_stage_forward(
    y: np.ndarray,
    base: np.ndarray,
    params: Dict[str, Any],
    *,
    bivariate_forward: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary_stages: list[UnaryFns],
) -> np.ndarray:
    t = bivariate_forward(y, base, params["bivariate_params"])
    for (_fit, forward_fn, _inv), p in zip(unary_stages, params["unary_stage_params"]):
        t = forward_fn(t, p)
    return t


def chain_multi_stage_inverse(
    t_final: np.ndarray,
    base: np.ndarray,
    params: Dict[str, Any],
    *,
    bivariate_inverse: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    unary_stages: list[UnaryFns],
) -> np.ndarray:
    t = t_final
    # Unary stages: reverse order at inverse time.
    for (_fit, _forward, inv_fn), p in zip(
        reversed(unary_stages), reversed(params["unary_stage_params"]),
    ):
        t = inv_fn(t, p)
    return bivariate_inverse(t, base, params["bivariate_params"])


__all__ = [
    "cbrt_y_fit", "cbrt_y_forward", "cbrt_y_inverse", "cbrt_y_domain",
    "log_y_fit", "log_y_forward", "log_y_inverse", "log_y_domain",
    "yeo_johnson_y_fit", "yeo_johnson_y_forward", "yeo_johnson_y_inverse", "yeo_johnson_y_domain",
    "quantile_normal_y_fit", "quantile_normal_y_forward", "quantile_normal_y_inverse", "quantile_normal_y_domain",
    "chain_bivariate_then_unary_fit",
    "chain_bivariate_then_unary_forward",
    "chain_bivariate_then_unary_inverse",
    "chain_multi_stage_fit",
    "chain_multi_stage_forward",
    "chain_multi_stage_inverse",
]
