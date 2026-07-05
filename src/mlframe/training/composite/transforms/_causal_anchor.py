"""``causal_anchor_residual`` composite transform.

``T = y - alpha*base``, inverse ``y = T_hat + alpha*base``, where ``alpha`` is a
SHRINK coefficient robustly fitted on ``(y, base)`` and CONSTRAINED to ``[0, 1]``.

Motivation. When ``base`` is a NOISY causal anchor (a rolling / EWMA central
value), ``diff`` commits fully (implicit ``alpha=1``) and over-shoots, while a
free ``linear_residual`` can fit a fragile large ``alpha`` that extrapolates
badly once the anchor drifts out of its train range on an unseen group. The
``[0, 1]`` clamp is the whole point: ``alpha<=1`` cannot over-commit past the
anchor and ``alpha>=0`` cannot invert its sign, so the additive inverse
``T_hat + alpha*base`` stays bounded relative to the anchor even when ``base``
is far out of range. The inverse is pure-additive (no ``beta``), keeping the
composite MLP-friendly like ``additive_residual`` -- any ``T_hat`` extrapolation
maps back to ``y`` by a single bounded addition.

Fit. A two-pass trimmed-LS (MAD) slope estimate (same robustness as
``linear_residual_robust``) gives ``alpha_raw``; scarce / uninformative data is
pulled toward a moderate prior of ``0.5`` by a sample-size blend
(``w = n/(n+n0)``); the result is clipped to ``[0, 1]``. The intercept from the
OLS is used ONLY to compute residuals for the robust trim -- it is discarded, so
the transform has a single fitted parameter ``alpha`` (the ``[0, 1]`` clamp IS
the parameter bound).

cProfile (see ``_benchmarks/bench_causal_anchor_fit.py``). At the representative
fit shape (n=200k) the fit is dominated by the two ``np.median`` sorts inside
the MAD trim; forward / inverse are single fused AXPY passes. An ``@njit``
rewrite of the robust-slope loop was benched against the numpy path: no
actionable speedup (numpy's C ``median`` / masked reductions already saturate
memory bandwidth; numba's median re-sorts identically), so the numpy path is the
default. Forward + inverse are already at the vectorised floor.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Moderate shrink prior: when data is scarce or the anchor carries no slope
# information, default alpha toward a half-way commit rather than 0 (ignore the
# anchor) or 1 (fully commit like diff).
_CAUSAL_ANCHOR_ALPHA_PRIOR: float = 0.5

# Effective sample size of the prior. alpha blends as w*alpha_raw + (1-w)*prior
# with w = n/(n+n0); n0=20 makes the prior negligible for the usual large-n fit
# but meaningfully regularises a tiny train fold toward the moderate shrink.
_CAUSAL_ANCHOR_PRIOR_PSEUDOCOUNT: float = 20.0

# Robust trim threshold (sigma-equivalent via MAD*1.4826); matches
# linear_residual_robust so the two share a tuned breakdown behaviour.
_CAUSAL_ANCHOR_MAD_K: float = 3.0

# Below this inlier fraction the MAD trim is considered destructive (all-outlier
# / degenerate scale) and the first-pass OLS slope is kept instead.
_CAUSAL_ANCHOR_MIN_KEEP_FRAC: float = 0.5

# Below this many finite rows the slope is uninformative -> return the prior.
_CAUSAL_ANCHOR_MIN_N: int = 5


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form single-base OLS ``y ~ slope*x + intercept``.

    Degenerate guards mirror ``_linear_residual_fit``: ``n<2`` -> ``(0, mean y)``
    (``(0, 0)`` for empty), zero-variance ``x`` -> ``(0, mean y)``.
    """
    n = x.size
    if n < 2:
        return 0.0, (float(np.mean(y)) if n > 0 else 0.0)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    dx = x - mx
    vx = float(np.dot(dx, dx))
    if vx <= 0.0:
        return 0.0, my
    slope = float(np.dot(dx, y - my) / vx)
    intercept = my - slope * mx
    return slope, intercept


def _robust_slope(base_f: np.ndarray, y_f: np.ndarray) -> float:
    """Two-pass trimmed-LS slope: OLS -> drop ``|resid|>k*sigma_MAD`` -> refit.

    Returns only the slope (the intercept is a nuisance used to centre the
    residuals for the trim). Falls back to the first-pass slope when the trim
    would drop no rows, drop too many, or the residual scale is degenerate.
    """
    n = base_f.size
    alpha1, beta1 = _ols_slope_intercept(base_f, y_f)
    resid = y_f - alpha1 * base_f - beta1
    if not np.all(np.isfinite(resid)):
        return alpha1
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_mad = mad * 1.4826
    if sigma_mad <= 0.0 or not np.isfinite(sigma_mad):
        return alpha1
    keep = np.abs(resid - med) <= _CAUSAL_ANCHOR_MAD_K * sigma_mad
    n_kept = int(keep.sum())
    if n_kept < max(2, int(_CAUSAL_ANCHOR_MIN_KEEP_FRAC * n)) or n_kept == n:
        return alpha1
    alpha2, _ = _ols_slope_intercept(base_f[keep], y_f[keep])
    return alpha2


def _causal_anchor_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,  # noqa: ARG001 - API symmetry; median-based fit ignores weights
) -> dict[str, Any]:
    """Fit the ``[0, 1]``-clamped anchor-shrink coefficient ``alpha``.

    ``sample_weight`` is accepted for registry-signature symmetry but ignored:
    weighting a MAD trim / robust slope is ill-defined for small clusters, same
    stance as ``theilsen_residual``.
    """
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if not bool(finite.all()):
        y_f = y_f[finite]
        base_f = base_f[finite]
    n = y_f.size
    if n < _CAUSAL_ANCHOR_MIN_N:
        return {"alpha": float(np.clip(_CAUSAL_ANCHOR_ALPHA_PRIOR, 0.0, 1.0))}

    alpha_raw = _robust_slope(base_f, y_f)
    if not np.isfinite(alpha_raw):
        alpha_raw = _CAUSAL_ANCHOR_ALPHA_PRIOR
    # Scarce-data shrink toward the moderate prior BEFORE the clamp: large n
    # keeps alpha_raw intact, tiny n pulls it toward 0.5.
    w = n / (n + _CAUSAL_ANCHOR_PRIOR_PSEUDOCOUNT)
    alpha_shrunk = w * alpha_raw + (1.0 - w) * _CAUSAL_ANCHOR_ALPHA_PRIOR
    alpha = float(np.clip(alpha_shrunk, 0.0, 1.0))
    return {"alpha": alpha}


def _causal_anchor_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    return y - alpha * base


def _causal_anchor_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    return t_hat + alpha * base


def _causal_anchor_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)
