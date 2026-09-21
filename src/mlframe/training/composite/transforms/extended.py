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

from ._domain_shared import residual_domain_plain

import logging
from typing import Any, Callable, Optional, cast

import numpy as np

logger = logging.getLogger("mlframe.training.composite_transforms_extended")


_RECIPROCAL_EPS_FLOOR: float = 1e-12
_RECIPROCAL_Y_CAP_MULT: float = 1e3  # the inverse's z-floor bounds |y_hat| at this multiple of the train max|y|
_SPLINE_DEFAULT_K: int = 3
_SPLINE_DEFAULT_S_MULT: float = 1.0  # smoothing = m * var_noise * s_mult (scipy s bounds the residual SS)


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
    """OLS-fit alpha/beta of arcsinh(y) ~ alpha * arcsinh(base) + beta on finite train rows; falls back to alpha=1/0 pass-through when fewer than 10 finite rows or base has zero variance."""
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
    """Apply the fitted asinh residual: arcsinh(y) - alpha * arcsinh(base) - beta."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return np.arcsinh(np.asarray(y, dtype=np.float64)) - alpha * np.arcsinh(np.asarray(base, dtype=np.float64)) - beta


def _asinh_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the asinh residual back to y-space: sinh(t_hat + alpha * arcsinh(base) + beta)."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    z = np.asarray(t_hat, dtype=np.float64) + alpha * np.arcsinh(np.asarray(base, dtype=np.float64)) + beta
    return np.sinh(z)


_asinh_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_plain


# ============================================================
# 2. centered_ratio: T = y / (base + c)
# ============================================================
# Extension of ratio to signed bases. ``c`` is fitted at train-time
# so (base + c) is strictly positive on train (subject to eps floor).

def _centered_ratio_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """Fit a shift ``c`` so ``base + c`` is strictly positive on train, plus an eps floor scaled to the median absolute base value.

    A strictly positive base needs no shift (``c = 0``, the transform is then ``ratio``): shifting it so the train minimum lands at ~0 put a pole just
    below the train range, so a predict base a few percent under the train minimum flipped the sign of ``y_hat``. A base that reaches zero or goes
    negative is shifted by a margin proportional to its spread (the larger of the median |base| and the IQR), not a 1% sliver, so the train minimum
    sits a full scale unit away from the pole.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(base_arr)
    if not finite.any():
        return {"c": 0.0, "eps": _RECIPROCAL_EPS_FLOOR}
    b_fin = base_arr[finite]
    b_min = float(b_fin.min())
    b_scale = float(np.median(np.abs(b_fin)))
    eps = max(b_scale * 1e-6, _RECIPROCAL_EPS_FLOOR)
    if b_min > eps:
        return {"c": 0.0, "eps": eps}
    q25, q75 = np.percentile(b_fin, [25, 75])
    margin = max(b_scale, float(q75 - q25))
    if not margin > 0.0:
        margin = max(abs(b_min), 1.0)
    c = -b_min + margin
    return {"c": float(c), "eps": eps}


def _centered_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Compute ``y / (base + c)`` with the shifted denominator eps-floored away from zero."""
    c = float(params["c"])
    eps = float(params["eps"])
    shifted = np.asarray(base, dtype=np.float64) + c
    safe = np.where(np.abs(shifted) < eps, np.sign(shifted + 1e-300) * eps, shifted)
    return np.asarray(np.asarray(y, dtype=np.float64) / safe)


def _centered_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert centered_ratio: y = t_hat * (base + c), using the same eps-floored shifted base as forward."""
    c = float(params["c"])
    # Mirror the forward eps-floor on (base + c) so the round-trip stays exact on near-zero shifted rows (forward divides by the floored value, not the raw one).
    eps = float(params["eps"])
    shifted = np.asarray(base, dtype=np.float64) + c
    safe = np.where(np.abs(shifted) < eps, np.sign(shifted + 1e-300) * eps, shifted)
    return np.asarray(np.asarray(t_hat, dtype=np.float64) * safe)


_centered_ratio_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_plain


# ============================================================
# 3. polynomial_residual_deg2: T = y - a1*base - a2*base^2 - b
# ============================================================
# Degree-2 least squares on the centred/scaled (1, z, z^2) design.

def _polynomial_residual_deg2_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """Fit y ~ g0 + g1*z + g2*z^2 with ``z = (base - center) / scale`` by least squares (QR/SVD, not normal equations), then store both the
    z-space coefficients the forward/inverse evaluate and the equivalent raw-base ``(alpha1, alpha2, beta)`` for reporting.

    Solving the raw ``[1, b, b^2]`` normal equations squares an already huge condition number once the base is offset from zero (years, prices,
    timestamps), leaving about a third of the signal in T. Centring and scaling first keeps the design well conditioned at any offset, and
    evaluating in z space keeps the forward free of the ``alpha2*b^2`` cancellation. Too few finite rows fall back to ``T = y - mean(y)``.
    """
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.isfinite(bv)
    if finite.sum() < 10:
        y_mean = float(yv[finite].mean()) if finite.any() else 0.0
        return {"alpha1": 0.0, "alpha2": 0.0, "beta": y_mean, "b_center": 0.0, "b_scale": 1.0, "g0": y_mean, "g1": 0.0, "g2": 0.0}
    yc = yv[finite]
    bc = bv[finite]
    center = float(bc.mean())
    scale = float(bc.std())
    if not (np.isfinite(scale) and scale > 0.0):
        y_mean = float(yc.mean())
        return {"alpha1": 0.0, "alpha2": 0.0, "beta": y_mean, "b_center": center, "b_scale": 1.0, "g0": y_mean, "g1": 0.0, "g2": 0.0}
    z = (bc - center) / scale
    X = np.column_stack([np.ones_like(z), z, z * z])
    coef, *_ = np.linalg.lstsq(X, yc, rcond=None)
    g0, g1, g2 = (float(c) for c in coef)
    # Raw-base coefficients of the same polynomial, for provenance / gates that read alpha1/alpha2/beta.
    alpha2 = g2 / (scale * scale)
    alpha1 = g1 / scale - 2.0 * g2 * center / (scale * scale)
    beta = g0 - g1 * center / scale + g2 * center * center / (scale * scale)
    return {
        "beta": float(beta), "alpha1": float(alpha1), "alpha2": float(alpha2),
        "b_center": center, "b_scale": scale, "g0": g0, "g1": g1, "g2": g2,
    }


def _polynomial_residual_deg2_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Evaluate the fitted quadratic at ``base``: in centred z space when the params carry it, else (params pickled before centring) on the raw base."""
    bv = np.asarray(base, dtype=np.float64)
    if "b_center" in params:
        z = (bv - float(params["b_center"])) / float(params["b_scale"])
        return float(params["g0"]) + float(params["g1"]) * z + float(params["g2"]) * z * z
    return float(params["alpha1"]) * bv + float(params["alpha2"]) * bv * bv + float(params["beta"])


def _polynomial_residual_deg2_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Compute the degree-2 residual y - (alpha1*base + alpha2*base^2 + beta)."""
    return np.asarray(np.asarray(y, dtype=np.float64) - _polynomial_residual_deg2_g(base, params))


def _polynomial_residual_deg2_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the degree-2 residual: y = t_hat + alpha1*base + alpha2*base^2 + beta."""
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + _polynomial_residual_deg2_g(base, params))


_polynomial_residual_deg2_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_plain


# ============================================================
# 4. rank_residual: T = rank(y)/n - alpha * rank(base)/n
# ============================================================
# Distribution-free monotone residual. Forward maps via the
# train-fitted rank-to-value knot tables (one per axis). Inverse reads
# the y table backwards to recover y.

_RANK_MAX_KNOTS: int = 2048
"""Cap on the rank-to-value knots ``rank_residual`` stores per axis. The params used to carry both full sorted train arrays (O(n): 160 MB at 10M
rows, pickled into every saved model); a piecewise-linear knot table of this size reproduces the train rank map to within one knot spacing and
keeps the round trip exact on every train row (linear interpolation is inverted exactly by the reverse interpolation)."""


def _rank_knots(sorted_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strictly increasing ``(values, rank_fraction)`` knots of a sorted sample: each unique value at its mid-rank ``(first + last + 1) / (2n)``,
    thinned to at most ``_RANK_MAX_KNOTS`` knots (both extremes always kept)."""
    n = sorted_x.size
    uniq, first = np.unique(sorted_x, return_index=True)
    last = np.append(first[1:], n) - 1
    q = (first + last + 1).astype(np.float64) / (2.0 * n)
    if uniq.size > _RANK_MAX_KNOTS:
        keep = np.unique(np.linspace(0, uniq.size - 1, _RANK_MAX_KNOTS).round().astype(np.int64))
        uniq, q = uniq[keep], q[keep]
    if uniq.size == 1:
        # Constant axis: a zero-width ramp would divide by zero in interp; a scale-relative sliver keeps it invertible.
        span = max(abs(float(uniq[0])) * 1e-12, 1e-300)
        return np.array([uniq[0], uniq[0] + span]), np.array([q[0], q[0] + 1e-12])
    return uniq.astype(np.float64), q


def _rank_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """Fit the rank-space residual: OLS of empirical-CDF rank(y) on rank(base), storing bounded (value, rank-fraction) knot tables for each axis
    that reproduce the rank map and invert it back to y-space."""
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(yv) & np.isfinite(bv)
    if finite.sum() < 10:
        return {
            "y_knots": np.array([0.0, 1.0]), "y_q": np.array([0.0, 1.0]),
            "b_knots": np.array([0.0, 1.0]), "b_q": np.array([0.0, 1.0]),
            "alpha": 0.0,
            "beta": 0.5,
        }
    yc = yv[finite]
    bc = bv[finite]
    y_knots, y_q = _rank_knots(np.sort(yc))
    b_knots, b_q = _rank_knots(np.sort(bc))
    yr = np.interp(yc, y_knots, y_q)
    br = np.interp(bc, b_knots, b_q)
    b_mean = float(br.mean())
    bc_ranks = br - b_mean
    denom = float(np.dot(bc_ranks, bc_ranks))
    out: dict[str, Any] = {"y_knots": y_knots, "y_q": y_q, "b_knots": b_knots, "b_q": b_q}
    if denom <= 0:
        out.update(alpha=0.0, beta=float(yr.mean()))
        return out
    alpha = float(np.dot(bc_ranks, yr - yr.mean()) / denom)
    out.update(alpha=alpha, beta=float(yr.mean() - alpha * b_mean))
    return out


def _rank_residual_ranks(x: np.ndarray, params: dict[str, Any], axis: str) -> np.ndarray:
    """Train-calibrated rank fraction of ``x`` on axis ``"y"`` or ``"b"``; params pickled before the knot tables keep their sorted-array lookup."""
    xv = np.asarray(x, dtype=np.float64)
    if f"{axis}_knots" in params:
        return np.interp(xv, np.asarray(params[f"{axis}_knots"], dtype=np.float64), np.asarray(params[f"{axis}_q"], dtype=np.float64))
    srt = np.asarray(params[f"{axis}_sorted"], dtype=np.float64)
    return (np.searchsorted(srt, xv, side="left").astype(np.float64) + 0.5) / max(srt.size, 1)


def _rank_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Map y and base to train-calibrated rank fractions and return the residual rank(y) - alpha * rank(base) - beta."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return _rank_residual_ranks(y, params, "y") - alpha * _rank_residual_ranks(base, params, "b") - beta


def _rank_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the rank residual by recovering the predicted y-rank fraction and reading the train rank-to-value map backwards; a rank outside the
    train knot range clamps to the train y-range."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    yr_hat = np.asarray(t_hat, dtype=np.float64) + alpha * _rank_residual_ranks(base, params, "b") + beta
    if "y_knots" in params:
        return np.asarray(np.interp(yr_hat, np.asarray(params["y_q"], dtype=np.float64), np.asarray(params["y_knots"], dtype=np.float64)))
    y_sorted = np.asarray(params["y_sorted"], dtype=np.float64)
    n_y = y_sorted.size
    if n_y < 2:
        return np.full(yr_hat.shape, float(y_sorted[0]) if n_y else 0.0)
    yr_clipped = np.clip(yr_hat, 0.0, 1.0 - 1e-9)
    idx = np.clip((yr_clipped * n_y).astype(np.int64), 0, n_y - 1)
    return y_sorted[idx]


_rank_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_plain


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
    """Fit a smoothing-spline mean curve g(base) on train (base, y): bucket y by unique base value, estimate residual noise variance via the lag-1 first-difference (Rice) estimator on the bucket means, and set scipy's ``s`` smoothing factor to m * noise_var so the spline absorbs noise, not signal. Falls back to a flat y-mean curve when fewer than 20 finite rows or fewer than 4 unique base values."""
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
    # scipy UnivariateSpline's smoothing factor bounds the residual sum of
    # squares: sum((y - g(b))^2) <= s, so the correct scale is m * var(noise),
    # NOT m * std(signal). The previous m * std(yc_avg) over-smoothed badly
    # (std not variance, and signal std not noise std) -- the spline absorbed
    # the genuine signal into g(base), leaving residual ~ signal instead of
    # noise. Estimate the noise variance robustly with the lag-1 first-
    # difference (Rice / von Neumann) estimator on the base-sorted means:
    # consecutive differences of a smooth curve are noise-dominated, so
    # 0.5 * mean(diff^2) is a signal-curvature-robust noise-variance estimate.
    if yc_avg.size >= 2:
        d = np.diff(yc_avg)
        var_noise = 0.5 * float(np.mean(d * d))
    else:
        var_noise = float(np.var(yc_avg))
    s = max(unique_b.size, 1) * var_noise * _SPLINE_DEFAULT_S_MULT
    params: dict[str, Any] = {
        "knots_b": unique_b.astype(np.float64, copy=False),
        "knots_y": yc_avg.astype(np.float64, copy=False),
        "s": float(s),
        "y_mean": float(yc.mean()),
    }
    # Build the spline once at fit: a construction failure would otherwise turn the transform into ``T = y - mean(y)`` at every forward/inverse
    # call with nothing but a log line. Flag it so discovery rejects the spec instead of training on a silently degraded target.
    spl = _build_smoothing_spline(params)
    params["is_degenerate"] = spl is None
    if spl is not None:
        # Keep the fitted B-spline (a handful of knots) instead of every unique train base: the raw tables grew O(n) in the pickled params
        # (3.2 MB at 1e5 rows) and the spline was refitted on every forward/inverse call.
        t, c, k = spl._eval_args
        params.update(knots_b=np.zeros(0, dtype=np.float64), knots_y=np.zeros(0, dtype=np.float64),
                      tck_t=np.asarray(t, dtype=np.float64), tck_c=np.asarray(c, dtype=np.float64), tck_k=int(k))
    return params


class _TckSpline:
    """A fitted B-spline evaluated with constant extrapolation, as ``UnivariateSpline(ext="const")`` does."""

    def __init__(self, t: np.ndarray, c: np.ndarray, k: int):
        self.tck = (t, c, k)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.interpolate import splev
        return cast(np.ndarray, splev(x, self.tck, ext=3))


def _build_smoothing_spline(params: dict[str, Any]) -> Any:
    """Rebuild the ``UnivariateSpline`` from the stored knots, or ``None`` (logged at WARNING, throttled) when scipy cannot build it."""
    from mlframe.utils.log_throttle import log_throttle
    if "tck_t" in params:
        return _TckSpline(np.asarray(params["tck_t"], dtype=np.float64), np.asarray(params["tck_c"], dtype=np.float64), int(params["tck_k"]))
    knots_b = np.asarray(params.get("knots_b", []), dtype=np.float64)
    knots_y = np.asarray(params.get("knots_y", []), dtype=np.float64)
    if knots_b.size < 4:
        return None
    try:
        from scipy.interpolate import UnivariateSpline
        return UnivariateSpline(knots_b, knots_y, k=_SPLINE_DEFAULT_K, s=float(params.get("s", 0.0)), ext="const")
    except Exception as exc:
        log_throttle(
            logger, "smoothing_spline_build_failed", logging.WARNING,
            "smoothing_spline_residual: UnivariateSpline build failed (%s: %s); g(base) falls back to the train y mean, so T = y - mean(y).",
            type(exc).__name__, exc,
        )
        return None


def _smoothing_spline_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Rebuild the UnivariateSpline from the stored train knots and evaluate g(base); falls back to the constant train y-mean when there are too few knots or the spline construction/evaluation raises (logged at WARNING)."""
    from mlframe.utils.log_throttle import log_throttle
    bv = np.asarray(base, dtype=np.float64).reshape(-1)
    y_mean = float(params.get("y_mean", 0.0))
    spl = _build_smoothing_spline(params)
    if spl is None:
        return np.full(bv.shape, y_mean, dtype=np.float64)
    try:
        g = np.asarray(spl(bv), dtype=np.float64)
    except Exception as exc:
        log_throttle(
            logger, "smoothing_spline_eval_failed", logging.WARNING,
            "smoothing_spline_residual: spline evaluation failed (%s: %s); g(base) falls back to the train y mean.", type(exc).__name__, exc,
        )
        return np.full(bv.shape, y_mean, dtype=np.float64)
    return np.where(np.isfinite(g), g, y_mean)


def _smoothing_spline_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Compute the residual y - g(base) using the fitted smoothing spline."""
    return np.asarray(np.asarray(y, dtype=np.float64) - _smoothing_spline_g(base, params))


def _smoothing_spline_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the spline residual: y = t_hat + g(base)."""
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + _smoothing_spline_g(base, params))


_smoothing_spline_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_plain


# ============================================================
# 6. reciprocal_residual: T = 1/y - 1/base
# ============================================================
# Niche: useful when y has multiplicative jump dynamics.

def _reciprocal_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """Derive eps floors for y and base (scaled to each column's median absolute value) used to guard the 1/x terms against division blow-up near zero,
    plus the inverse's floor on ``z = T_hat + 1/base``, which lives in 1/y units and so is derived from the y range, never from the base scale."""
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(base, dtype=np.float64)
    y_fin = yv[np.isfinite(yv)]
    y_scale = float(np.median(np.abs(y_fin))) if y_fin.size else 1.0
    b_scale = float(np.median(np.abs(bv[np.isfinite(bv)]))) if np.isfinite(bv).any() else 1.0
    y_absmax = float(np.max(np.abs(y_fin))) if y_fin.size else 1.0
    return {
        "eps_y": max(y_scale * 1e-6, _RECIPROCAL_EPS_FLOOR),
        "eps_b": max(b_scale * 1e-6, _RECIPROCAL_EPS_FLOOR),
        "eps_z": _reciprocal_eps_z(y_absmax),
    }


def _reciprocal_eps_z(y_absmax: float) -> float:
    """Floor on ``|z|`` (z ~ 1/y) that caps ``|y_hat|`` at ``_RECIPROCAL_Y_CAP_MULT`` times the train ``max|y|``; every train row has ``|z| >= 1/max|y|`` so the floor never touches them."""
    ref = y_absmax if np.isfinite(y_absmax) and y_absmax > 0 else 1.0
    return max(1.0 / (_RECIPROCAL_Y_CAP_MULT * ref), 1e-300)


def _reciprocal_resolve_eps_z(params: dict[str, Any]) -> float:
    """``eps_z`` from params; params pickled before it existed derive it from ``eps_y`` (= 1e-6 * median|y|), so old models get the y-unit floor too."""
    if "eps_z" in params:
        return float(params["eps_z"])
    eps_y = float(params.get("eps_y", 1e-6))
    return _reciprocal_eps_z(eps_y / 1e-6)


def _reciprocal_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Compute T = 1/y - 1/base with both denominators eps-floored away from zero."""
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
    """Invert the reciprocal residual: z = t_hat + 1/base, then y = 1/z, with z eps-floored so a near-zero z does not blow up the prediction."""
    eps_b = float(params["eps_b"])
    bv = np.asarray(base, dtype=np.float64)
    safe_b = np.where(np.abs(bv) < eps_b, np.sign(bv + 1e-300) * eps_b, bv)
    z = np.asarray(t_hat, dtype=np.float64) + 1.0 / safe_b
    # y = 1 / z; a near-zero z would blow the prediction up. The floor is in z (1/y) units: a base-unit eps here collapsed every
    # prediction to the constant 1/eps_b once |y| exceeded ~1e6/median|base|.
    eps_z = _reciprocal_resolve_eps_z(params)
    safe_z = np.where(np.abs(z) < eps_z, np.sign(z + 1e-300) * eps_z, z)
    return 1.0 / safe_z


def _reciprocal_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Row mask for reciprocal_residual: base (and y, when supplied) must be finite and strictly nonzero, since both feed a 1/x term."""
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.isfinite(bv) & (np.abs(bv) > 0)
    if y is None:
        return base_ok
    yv = np.asarray(y, dtype=np.float64)
    return np.asarray(base_ok & np.isfinite(yv) & (np.abs(yv) > 0))


# ============================================================
# 7. geometric_mean_residual: T = y / geomean(bases)
# ============================================================
# Multi-base, multiplicative. Requires every base column > 0 on
# the row (strict positivity). Inverse: y = T * geomean(bases).

def _geometric_mean_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """Derive an eps floor (scaled to the median absolute base value) for the log-mean-exp geomean computation, and record the number of base columns."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    scale = float(np.median(np.abs(bv[np.isfinite(bv)]))) if np.isfinite(bv).any() else 1.0
    return {"eps": max(scale * 1e-6, _RECIPROCAL_EPS_FLOOR), "n_bases": int(bv.shape[1])}


def _geometric_mean_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Compute T = y / geomean(bases), with the geomean via log-mean-exp and both the per-column base and the resulting geomean eps-floored."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    eps = float(params["eps"])
    bv = np.asarray(base, dtype=np.float64)
    # geomean via log-mean-exp. Strict positivity required by domain.
    log_b = np.log(np.where(bv > eps, bv, eps))
    g = np.exp(log_b.mean(axis=1))
    return np.asarray(np.asarray(y, dtype=np.float64) / np.where(g > eps, g, eps))


def _geometric_mean_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the geometric-mean residual: y = t_hat * geomean(bases)."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    eps = float(params["eps"])
    bv = np.asarray(base, dtype=np.float64)
    log_b = np.log(np.where(bv > eps, bv, eps))
    g = np.exp(log_b.mean(axis=1))
    # Mirror the forward's floor on the geomean so the round trip is exact on rows where it binds.
    return np.asarray(np.asarray(t_hat, dtype=np.float64) * np.where(g > eps, g, eps))


def _geometric_mean_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Row mask for geometric_mean_residual: every base column on the row must be finite and strictly positive (required for the log-mean-exp
    geomean); y only needs to be finite, since ``T = y / geomean`` and its inverse are a plain division and multiplication defined for any real y."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.all(np.isfinite(bv) & (bv > 0), axis=1)
    if y is None:
        return np.asarray(base_ok)
    yv = np.asarray(y, dtype=np.float64)
    return np.asarray(base_ok & np.isfinite(yv))


# ============================================================
# 8. pairwise_interaction_residual: T = y - alpha*prod(bases) - beta
# ============================================================
# Multi-base, BILINEAR/multilinear. Captures pure interaction term;
# residual after removing the multiplicative contribution. ``alpha``
# fitted via OLS on (1, prod(bases)) train pairs.

def _pairwise_interaction_residual_fit(
    y: np.ndarray, base: np.ndarray,
) -> dict[str, Any]:
    """OLS-fit y ~ alpha * prod(bases) + beta on finite train rows, capturing the pure multiplicative interaction term across all base columns."""
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
    """Compute T = y - alpha * prod(bases) - beta."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    p = np.prod(bv, axis=1)
    return np.asarray(np.asarray(y, dtype=np.float64) - alpha * p - beta)


def _pairwise_interaction_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert the pairwise interaction residual: y = t_hat + alpha * prod(bases) + beta."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    bv = np.asarray(base, dtype=np.float64)
    p = np.prod(bv, axis=1)
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + alpha * p + beta)


def _pairwise_interaction_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Row mask for pairwise_interaction_residual: every base column must be finite; no positivity constraint since the interaction is a plain product/residual, not a ratio or log."""
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    bv = np.asarray(base, dtype=np.float64)
    base_ok = np.all(np.isfinite(bv), axis=1)
    if y is None:
        return np.asarray(base_ok)
    yv = np.asarray(y, dtype=np.float64)
    return np.asarray(base_ok & np.isfinite(yv))


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
