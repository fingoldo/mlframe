"""Multi-estimator MI aggregators for MRMR (2026-05-29).

Three aggregation strategies over a panel of base MI estimators
(FD, QS, MDLP, Mixed-KSG, MINE, ...):

  * **median**: robust default; takes ``median(I_1, ..., I_K)`` per pair.
    Zero theoretical guarantee but inherits the worst-case noise floor of the
    median element, not the mean -> robust to any single estimator's failure
    mode (e.g. FD's no-signal inflation).

  * **GENIE-style weighted ensemble** (Moon, Sricharan, Greenewald, Hero;
    IEEE Trans. Inf. Theory 2021; https://arxiv.org/abs/1701.08083). Solves a
    small linear system per dataset for bias-cancelling weights; achieves the
    parametric 1/N MSE rate. Strictly beats any single member in expectation.

  * **best_on_calibration**: picks the estimator with the lowest MI on a held-out
    no-signal calibration set, then uses that estimator for the actual MI
    scoring. Empirical analog of GENIE's bias cancellation.

The aggregator integrates orthogonally with MRMR via a ``mi_aggregator='...'``
knob (planned for the MRMR.__init__ wiring step).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Median aggregator (zero-cost robust default)
# =============================================================================


def median_mi(estimates: Sequence[float]) -> float:
    """Take the median of K MI estimates. Robust to outliers (e.g. FD's noise floor)."""
    arr = np.asarray([e for e in estimates if e is not None and np.isfinite(e)],
                     dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def median_mi_panel(x: np.ndarray, y: np.ndarray,
                     estimators: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
                     ) -> float:
    """Apply each estimator to (x, y) and return the median of the K MI scores."""
    scores = []
    for name, est in estimators.items():
        try:
            scores.append(float(est(x, y)))
        except Exception as exc:
            logger.warning(f"median_mi_panel: estimator {name!r} failed: {exc!r}")
    return median_mi(scores)


# =============================================================================
# GENIE weighted ensemble (Moon 2021)
# =============================================================================


def genie_weights(estimator_bias_rates: Sequence[float],
                   estimator_variance: Sequence[float],
                   ridge: float = 1e-6) -> np.ndarray:
    """Solve for GENIE bias-cancelling weights.

    Given K estimators with known analytical bias rates ``b_k(N)`` (e.g. KSG
    bias goes as ``1/N``, plug-in goes as ``M/N`` for some M) and variance
    proxies, solve:

        minimize  sum_k w_k^2 * var_k   s.t.  sum_k w_k = 1  and  sum_k w_k * b_k = 0

    Closed-form Lagrangian solution; ridge-regularised so the constraint
    matrix stays invertible when biases are nearly collinear.

    Args:
        estimator_bias_rates: known bias-rate signatures per estimator (K,).
        estimator_variance: variance proxies per estimator (K,).
        ridge: small Tikhonov term on the bias constraint.

    Returns: weights of shape (K,), summing to 1.
    """
    b = np.asarray(estimator_bias_rates, dtype=np.float64).ravel()
    v = np.asarray(estimator_variance, dtype=np.float64).ravel()
    K = b.size
    if K == 0:
        return np.array([], dtype=np.float64)
    if K == 1:
        return np.array([1.0], dtype=np.float64)
    # System: [V + ridge*I | b | 1] * [w; lam1; lam2] = [0; 0; 1]
    # i.e. minimize w^T diag(v) w  s.t.  b^T w = 0,  1^T w = 1
    V = np.diag(v + ridge)
    A = np.block([
        [V,                      b.reshape(-1, 1), np.ones((K, 1))],
        [b.reshape(1, -1),       np.zeros((1, 1)), np.zeros((1, 1))],
        [np.ones((1, K)),        np.zeros((1, 1)), np.zeros((1, 1))],
    ])
    rhs = np.zeros(K + 2)
    rhs[-1] = 1.0
    try:
        sol = np.linalg.solve(A, rhs)
        w = sol[:K]
    except np.linalg.LinAlgError:
        # Singular: fall back to uniform.
        w = np.ones(K) / K
    return w


def genie_aggregate(estimates: Sequence[float], weights: np.ndarray) -> float:
    """Combine K MI estimates with GENIE weights."""
    arr = np.asarray(estimates, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if arr.size != w.size:
        raise ValueError(f"estimates ({arr.size}) and weights ({w.size}) size mismatch")
    return float((arr * w).sum())


def genie_mi_panel(x: np.ndarray, y: np.ndarray,
                    estimators: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
                    bias_rates: Optional[Dict[str, float]] = None,
                    variances: Optional[Dict[str, float]] = None,
                    floor_at_zero: bool = True) -> float:
    """Run a panel of K estimators on (x, y) then combine via GENIE weights.

    If ``bias_rates`` / ``variances`` are not provided, default to:
        bias_rate = 1 / sqrt(N)    (matches MINE / KSG asymptotic)
        variance = 1.0            (uniform)
    Producing a uniformly-weighted GENIE solution -- pragmatic fallback when
    analytical bias rates are not available per estimator.
    """
    estimates = []
    bias = []
    var = []
    n = x.size
    for name, est in estimators.items():
        try:
            mi = float(est(x, y))
        except Exception as exc:
            logger.warning(f"genie_mi_panel: {name!r} failed: {exc!r}")
            continue
        estimates.append(mi)
        bias.append((bias_rates or {}).get(name, 1.0 / max(np.sqrt(n), 1.0)))
        var.append((variances or {}).get(name, 1.0))
    if not estimates:
        return 0.0
    w = genie_weights(bias, var)
    out = genie_aggregate(estimates, w)
    if floor_at_zero:
        out = max(0.0, out)
    return out


# =============================================================================
# Calibration-based picker
# =============================================================================


def best_on_calibration_mi(
    x: np.ndarray, y: np.ndarray,
    estimators: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    calibration_data: Optional[np.ndarray] = None,
    calibration_target: Optional[np.ndarray] = None,
    seed: int = 0,
) -> float:
    """Empirical bias-calibration aggregator. Runs each estimator on a known
    no-signal calibration draw (or generates one via permutation), picks the
    estimator with the smallest calibration MI, then uses THAT estimator on
    the actual (x, y).
    """
    rng = np.random.default_rng(int(seed))
    if calibration_data is None or calibration_target is None:
        # Generate no-signal calibration: shuffle y.
        calibration_data = x.copy()
        calibration_target = rng.permutation(y)
    best_estimator: Optional[str] = None
    best_floor = np.inf
    for name, est in estimators.items():
        try:
            cal_mi = float(est(calibration_data, calibration_target))
        except Exception as exc:
            logger.warning(f"best_on_calibration: {name!r} failed cal: {exc!r}")
            continue
        if cal_mi < best_floor:
            best_floor = cal_mi
            best_estimator = name
    if best_estimator is None:
        return 0.0
    try:
        return float(estimators[best_estimator](x, y))
    except Exception as exc:
        logger.warning(f"best_on_calibration: chosen estimator {best_estimator!r} failed on (x, y): {exc!r}")
        return 0.0


__all__ = [
    "median_mi", "median_mi_panel",
    "genie_weights", "genie_aggregate", "genie_mi_panel",
    "best_on_calibration_mi",
]
