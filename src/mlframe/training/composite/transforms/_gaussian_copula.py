"""``gaussian_copula_residual`` composite transform.

``T = z_y - alpha * z_b - beta`` where ``z_y = Phi^-1(ecdf_y(y))`` and ``z_b = Phi^-1(ecdf_base(base))`` are the normal scores of ``y`` / ``base`` under
their TRAIN empirical CDFs, and (alpha, beta) is the OLS line in normal-scores space. This is the Gaussian-copula view of the (y, base) dependence:
any monotone marginal distortion of either side collapses to the identity (like ``rank_ecdf_residual``), while the residual itself lives on a Gaussian
scale that RMSE-trained downstream models fit cleanly (unlike the bounded uniform rank residual). Inverse maps back through the stored y-ECDF knots:
``y_hat = quantile_y(Phi(T_hat + alpha * z_b + beta))``, so reconstructions cannot leave the train y-support (edge-knot clamping via ``np.interp``).

ECDF knots reuse the ``_rank_ecdf`` helper; the normal-score clip mirrors ``quantile_normal_y`` (u clipped to ``[eps, 1-eps]`` with eps from the knot
count so ``Phi^-1`` stays finite at the tails -- this makes the round-trip lossy only on the extreme tail rows, like ``quantile_normal_y``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._rank_ecdf import _ecdf_knots


def _copula_z(x: np.ndarray, knots: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Normal score of ``x`` under the fitted ECDF knots: ``Phi^-1(clip(ecdf(x), eps, 1-eps))``."""
    from scipy.special import ndtri
    u = np.interp(np.asarray(x, dtype=np.float64), knots, cdf)
    eps = 1.0 / (2.0 * max(len(cdf), 2))
    return np.asarray(ndtri(np.clip(u, eps, 1.0 - eps)))


def _gaussian_copula_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Store train ECDF knots for ``y`` and ``base`` plus the OLS (alpha, beta) of the normal-scores regression ``z_y ~ alpha * z_b + beta``."""
    y_knots, y_cdf = _ecdf_knots(y)
    base_knots, base_cdf = _ecdf_knots(base)
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    z_y = _copula_z(y_f[finite], y_knots, y_cdf)
    z_b = _copula_z(base_f[finite], base_knots, base_cdf)
    if z_y.size < 3:
        alpha, beta = 0.0, 0.0
    else:
        zb_mean = float(z_b.mean())
        zb_c = z_b - zb_mean
        denom = float(np.dot(zb_c, zb_c))
        if denom <= 0:
            alpha = 0.0
            beta = float(z_y.mean())
        else:
            alpha = float(np.dot(zb_c, z_y - z_y.mean()) / denom)
            beta = float(z_y.mean() - alpha * zb_mean)
    return {
        "y_knots": y_knots, "y_cdf": y_cdf,
        "base_knots": base_knots, "base_cdf": base_cdf,
        "alpha": alpha, "beta": beta,
    }


def _gaussian_copula_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = z_y - alpha * z_b - beta`` in normal-scores space."""
    z_y = _copula_z(np.asarray(y, dtype=np.float64).reshape(-1), np.asarray(params["y_knots"], dtype=np.float64), np.asarray(params["y_cdf"], dtype=np.float64))
    z_b = _copula_z(np.asarray(base, dtype=np.float64).reshape(-1), np.asarray(params["base_knots"], dtype=np.float64), np.asarray(params["base_cdf"], dtype=np.float64))
    return np.asarray(z_y - float(params["alpha"]) * z_b - float(params["beta"]))


def _gaussian_copula_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: recover ``z_y = T_hat + alpha * z_b + beta``, map through ``Phi`` to a rank, then read the train y-quantile function (edge-knot clamp keeps y_hat inside the train support)."""
    from scipy.special import ndtr
    z_b = _copula_z(np.asarray(base, dtype=np.float64).reshape(-1), np.asarray(params["base_knots"], dtype=np.float64), np.asarray(params["base_cdf"], dtype=np.float64))
    z_y = np.asarray(t_hat, dtype=np.float64).reshape(-1) + float(params["alpha"]) * z_b + float(params["beta"])
    u = ndtr(z_y)
    return np.asarray(np.interp(u, np.asarray(params["y_cdf"], dtype=np.float64), np.asarray(params["y_knots"], dtype=np.float64)))


def _gaussian_copula_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))
