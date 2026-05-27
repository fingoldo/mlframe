"""Objective functions for ranking feature subsets by their SHAP-coalition proxy.

The proxy *margin* of a subset ``S`` on row ``i`` is ``base[i] + sum_{j in S} phi[i, j]``. The brute-
force / heuristic search maintains this margin incrementally (cheap running sum); the objective here
is the O(n) reduction applied on top of that margin against the true target ``y``.

Critical correction over the original research kernel: that kernel scored ``sum |y in {0,1} - margin|``
i.e. MAE of a 0/1 label against a *log-odds margin* -- dimensionally incoherent. We instead map the
margin through ``sigmoid`` for classification and score with a proper loss (Brier / log-loss), and
score regression in target space (RMSE / MAE). The incremental-margin speed trick is independent of
the metric (the running sum is the margin; the loss is a separate pointwise reduction), so proper
metrics cost the same O(n) per subset.

All objectives are returned as a LOSS (lower = better) so every optimizer minimises uniformly.
AUC (ranking metric, not pointwise) is supported only on the Python path (needs a per-subset sort),
not inside the numba hot loop -- documented in ``METRIC_CODES``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Integer codes for the numba hot loop. AUC is intentionally absent (needs sort; use the Python path).
METRIC_CODES = {"mae": 0, "rmse": 1, "mse": 1, "brier": 2, "logloss": 3}

_REGRESSION_METRICS = {"mae", "rmse", "mse"}
_CLASSIFICATION_METRICS = {"brier", "logloss", "auc"}


def resolve_metric(classification: bool, metric: str | None) -> str:
    """Pick / validate the objective metric for the task. Defaults: brier (clf), rmse (reg)."""
    if metric is None:
        return "brier" if classification else "rmse"
    metric = metric.lower()
    if classification and metric not in _CLASSIFICATION_METRICS:
        raise ValueError(f"metric={metric!r} is not a classification objective ({sorted(_CLASSIFICATION_METRICS)})")
    if not classification and metric not in _REGRESSION_METRICS:
        raise ValueError(f"metric={metric!r} is not a regression objective ({sorted(_REGRESSION_METRICS)})")
    return metric


@njit(cache=True, fastmath=True)
def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


@njit(cache=True, fastmath=True)
def score_margin(margin: np.ndarray, y: np.ndarray, metric_code: int) -> float:
    """Pointwise loss of a proxy margin vs target. Lower is better.

    metric_code: 0=MAE, 1=MSE (RMSE rank-equivalent), 2=Brier(sigmoid), 3=log-loss(sigmoid).
    """
    n = margin.shape[0]
    s = 0.0
    if metric_code == 0:  # MAE on raw margin (regression)
        for i in range(n):
            d = y[i] - margin[i]
            s += d if d >= 0.0 else -d
        return s / n
    if metric_code == 1:  # MSE on raw margin (regression); sqrt is monotone so omitted for ranking
        for i in range(n):
            d = y[i] - margin[i]
            s += d * d
        return s / n
    if metric_code == 2:  # Brier: (sigmoid(margin) - y)^2
        for i in range(n):
            p = _sigmoid(margin[i])
            d = p - y[i]
            s += d * d
        return s / n
    # metric_code == 3: log-loss with probability clipping
    eps = 1e-7
    for i in range(n):
        p = _sigmoid(margin[i])
        if p < eps:
            p = eps
        elif p > 1.0 - eps:
            p = 1.0 - eps
        s += -(y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))
    return s / n


def coalition_margin(phi: np.ndarray, base: np.ndarray, feature_idx) -> np.ndarray:
    """``base + sum over selected features of phi`` -- the proxy margin for one subset (numpy path)."""
    idx = np.asarray(feature_idx, dtype=np.int64)
    if idx.size == 0:
        return base.copy()
    return base + phi[:, idx].sum(axis=1)


def proxy_loss(margin: np.ndarray, y: np.ndarray, metric: str) -> float:
    """Python-path loss (lower = better), supports AUC (via mlframe.metrics.core) unlike the njit path."""
    metric = metric.lower()
    if metric == "auc":
        from mlframe.metrics.core import fast_roc_auc

        p = 1.0 / (1.0 + np.exp(-margin))
        auc = float(fast_roc_auc(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)))
        if not np.isfinite(auc):  # single-class slice -> uninformative; worst loss
            return 1.0
        return 1.0 - auc
    if metric == "rmse":
        return float(np.sqrt(score_margin(np.asarray(margin, np.float64), np.asarray(y, np.float64), 1)))
    code = METRIC_CODES[metric]
    return float(score_margin(np.asarray(margin, np.float64), np.asarray(y, np.float64), code))


def subset_loss(phi: np.ndarray, base: np.ndarray, y: np.ndarray, feature_idx, metric: str) -> float:
    """Convenience: coalition margin of a subset then its loss. Used by heuristics / re-validation."""
    return proxy_loss(coalition_margin(phi, base, feature_idx), y, metric)
