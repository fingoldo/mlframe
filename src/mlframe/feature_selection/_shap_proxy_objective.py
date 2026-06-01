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
from numba import njit, prange

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


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    # ``inline="always"`` ensures numba folds the branched-stable form into the per-element loops in
    # ``score_margin`` rather than emitting a function call (matters because this is the inner
    # hot path of Brier / log-loss scoring over hundreds of thousands of subsets per fit).
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


@njit(cache=True, fastmath=True, parallel=True)
def score_margin_parallel(margin: np.ndarray, y: np.ndarray, metric_code: int) -> float:
    """prange parallel twin of :func:`score_margin` for the tall-data Python hot path.

    Identical reduction, ``prange`` over rows so numba splits the per-element sigmoid+exp work
    across threads (the reduction variable ``s`` is a numba parallel-reduction, deterministic for
    a float sum up to summation-order round-off ~1e-15, which a LOSS used only for subset RANKING
    is insensitive to). Kept as a SEPARATE kernel from the serial one (never an in-place rewrite)
    because the serial loop is still called from the njit brute-force search kernels in
    ``_shap_proxy_search`` (which cannot dispatch at the Python level) and is faster below the
    crossover. Routed by ``score_margin_auto`` on row count; see that dispatcher for the measured
    crossover and the per-HW tuning hook."""
    n = margin.shape[0]
    s = 0.0
    if metric_code == 0:  # MAE
        for i in prange(n):
            d = y[i] - margin[i]
            s += d if d >= 0.0 else -d
        return s / n
    if metric_code == 1:  # MSE
        for i in prange(n):
            d = y[i] - margin[i]
            s += d * d
        return s / n
    if metric_code == 2:  # Brier
        for i in prange(n):
            p = _sigmoid(margin[i])
            d = p - y[i]
            s += d * d
        return s / n
    eps = 1e-7  # log-loss with probability clipping
    for i in prange(n):
        p = _sigmoid(margin[i])
        if p < eps:
            p = eps
        elif p > 1.0 - eps:
            p = 1.0 - eps
        s += -(y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))
    return s / n


# Row-count crossover above which the prange ``score_margin_parallel`` beats the serial loop on the
# Python hot path. Microbench (8-core, fastmath njit): brier 0.77x at n=5000 (parallel LOSES to
# thread-spawn overhead), 1.43x at n=10000, 2.19x at n=50000; logloss pays from n=5000 (1.92x) up
# (3.61x at n=100000). 10000 is the safe default: brier already wins there and logloss wins below it,
# so no caller regresses. Per-HW override via kernel_tuning_cache so the crossover tracks core count
# / exp throughput instead of a dev-box constant (mirrors the prefilter / cluster_su tuning hooks).
_SCORE_MARGIN_PARALLEL_MIN_ROWS = 10000
_score_margin_parallel_min_rows_cache: int | None = None


def _score_margin_parallel_min_rows() -> int:
    """Crossover row count for routing to ``score_margin_parallel``, from kernel_tuning_cache if set.

    Resolved once and memoized (the lookup may touch disk); falls back to the calibrated default on
    any error so a cache hiccup never changes the numeric result, only the serial/parallel route."""
    global _score_margin_parallel_min_rows_cache
    if _score_margin_parallel_min_rows_cache is not None:
        return _score_margin_parallel_min_rows_cache
    val = _SCORE_MARGIN_PARALLEL_MIN_ROWS
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_score_margin")
            if isinstance(entry, dict) and entry.get("parallel_min_rows"):
                val = int(entry["parallel_min_rows"])
    except Exception:
        pass
    _score_margin_parallel_min_rows_cache = val
    return val


def score_margin_auto(margin: np.ndarray, y: np.ndarray, metric_code: int) -> float:
    """Dispatch serial vs prange ``score_margin`` by row count (HW-tuned crossover).

    The fastest path is the default: tall margins (>= the crossover) take the parallel reduction,
    short ones (anchor subsets, small holdouts) stay serial to dodge thread-spawn overhead. Both
    kernels compute the same loss; the choice is purely a wall-clock route, never a semantic one."""
    if margin.shape[0] >= _score_margin_parallel_min_rows():
        return score_margin_parallel(margin, y, metric_code)
    return score_margin(margin, y, metric_code)


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
        return float(np.sqrt(score_margin_auto(np.asarray(margin, np.float64), np.asarray(y, np.float64), 1)))
    code = METRIC_CODES[metric]
    return float(score_margin_auto(np.asarray(margin, np.float64), np.asarray(y, np.float64), code))


def subset_loss(phi: np.ndarray, base: np.ndarray, y: np.ndarray, feature_idx, metric: str) -> float:
    """Convenience: coalition margin of a subset then its loss. Used by heuristics / re-validation."""
    return proxy_loss(coalition_margin(phi, base, feature_idx), y, metric)


def subset_uncertainty(phi_var: np.ndarray, feature_idx) -> float:
    """Attribution instability of a subset (lever #7): mean over rows of
    ``sqrt(sum_{j in S} Var_models(phi_j))``. Subsets whose SHAP attributions are unstable across the
    config-jittered models get a higher value -> can be penalised so the optimiser prefers subsets the
    proxy is confident about. Zero when no per-model variance was computed (n_models == 1)."""
    idx = np.asarray(feature_idx, dtype=np.int64)
    if phi_var is None or idx.size == 0:
        return 0.0
    return float(np.sqrt(phi_var[:, idx].sum(axis=1)).mean())
