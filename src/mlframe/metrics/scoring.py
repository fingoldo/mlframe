"""Scoring utilities salvaged from the legacy ``Models`` module.

Contains loss functions, scorers, a log-uniform distribution helper for
RandomizedSearchCV, and a proxy for scoring probabilistic classifier outputs.
"""

from __future__ import annotations


import numpy as np
import numba
from scipy.stats import uniform
from sklearn.metrics import make_scorer

from mlframe.metrics._numba_params import NUMBA_NJIT_PARAMS

# RMSE benefits from fastmath (associative reductions on float64) while the
# rest of mlframe.metrics keeps NUMBA_NJIT_PARAMS' fastmath=False default for
# bit-exact AUC / Brier scans. Override locally rather than mutate the shared
# dict.
_RMSE_NJIT_PARAMS = {**NUMBA_NJIT_PARAMS, "fastmath": True}


@numba.njit(**_RMSE_NJIT_PARAMS)
def _fast_rmse_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Single-pass RMSE: sum (y_true - y_pred)^2 / n then sqrt.

    Float64-only, contiguous-array fast path. Caller is responsible for the
    ``np.asarray(..., dtype=np.float64)`` cast when arrays might be other
    dtypes -- numba dispatches per dtype so a mixed-dtype call would
    re-compile, defeating the fastmath win.
    """
    n = y_true.shape[0]
    s = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return (s / n) ** 0.5


def fast_rmse(y_true, y_pred) -> float:
    """Numba single-pass RMSE. ~37x faster than np.sqrt(np.mean((y - p)**2))
    on n=100k float64 (889us -> 24us). Used by the honest-diagnostics
    bootstrap regression path where the inner loop calls RMSE 1000x per
    metric. Returns a Python float so it composes with the bootstrap
    aggregation's np.percentile + float() unchanged."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(_fast_rmse_kernel(y_true, y_pred))


def rmse_loss(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


rmse_score = make_scorer(rmse_loss, greater_is_better=False)


def rmsle_loss(y_true, y_pred):
    """Root mean squared logarithmic error. Negative predictions are clipped to 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None)), 2)))


rmsle_score = make_scorer(rmsle_loss, greater_is_better=False)


class log_uniform:
    """Log-uniform continuous distribution over ``[base**a, base**b]``.

    Compatible with scipy.stats random-variable interface expected by
    ``RandomizedSearchCV``.
    """

    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        myuniform = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, myuniform.rvs(random_state=random_state))
        return np.power(self.base, myuniform.rvs(size=size, random_state=random_state))


def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    """Wrap a scalar-target scorer so it can consume probability matrices.

    Passes column ``class_idx`` of ``y_probs`` to ``proxied_func(y_true, ...)``.
    """
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)
