"""Exact closed-form KNN-Shapley (Jia et al., VLDB 2019) -- the default data-valuation engine.

For a fixed validation point ``(x_val, y_val)`` and a K-nearest-neighbor classification utility, the
Shapley value of every training point has a recursive closed form computable in ``O(n log n)`` (the
sort dominates) with NO retraining: sort training points by distance to the validation point ascending
(rank 1 = nearest, rank n = farthest), then

    s_(n) = 1[y_(n) = y_val] / n
    s_(i) = s_(i+1) + (1[y_(i) = y_val] - 1[y_(i+1) = y_val]) / K * min(K, i) / i,  i = n-1 .. 1

computed from the farthest point (rank n) down to the nearest (rank 1). ``knn_shapley`` averages this
per-training-point value over all validation points. Classification only in v1 -- the closed form is
derived for a KNN classification agreement utility; regression callers should use ``tmc_shapley``
(``_mc_sampling.py``) instead, which is model-agnostic but requires retraining per permutation step.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.spatial.distance import cdist


@njit(cache=True, fastmath=True)
def _knn_shapley_recursion(match: np.ndarray, k: int) -> np.ndarray:
    """One validation point's per-training-point KNN-Shapley value, ``match`` sorted nearest(0)->farthest(n-1).

    ``match[j]`` is ``1.0`` if the training point at distance-rank ``j+1`` shares the validation point's
    label, else ``0.0``. Returns the same-length value array in the SAME (nearest-to-farthest) order.
    """
    n = match.shape[0]
    value = np.empty(n, dtype=np.float64)
    value[n - 1] = match[n - 1] / n
    for p in range(n - 2, -1, -1):
        i = p + 1  # 1-based rank of position p
        value[p] = value[p + 1] + (match[p] - match[p + 1]) / k * min(k, i) / i
    return value


def _score_one_batch(Xb: np.ndarray, yb: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Accumulate KNN-Shapley contributions of every training point over one batch of validation points."""
    n_train = X_train.shape[0]
    acc = np.zeros(n_train, dtype=np.float64)
    D = cdist(Xb, X_train, metric=metric)  # (batch, n_train)
    # One vectorized argsort over the whole (batch, n_train) matrix beats a Python loop of per-row
    # argsort calls -- same O(n log n) sort cost per row, but amortizes numpy's per-call dispatch
    # overhead across the batch (measured: this was the dominant cost at n_train=20000, ~0.9s of 1.1s).
    orders = np.argsort(D, axis=1, kind="stable")  # (batch, n_train), ascending distance per row
    for r in range(Xb.shape[0]):
        order = orders[r]
        match = (y_train[order] == yb[r]).astype(np.float64)
        acc[order] += _knn_shapley_recursion(match, k)
    return acc


def knn_shapley(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
    standardize: bool = True,
    n_jobs: int = -1,
    batch_val: int = 256,
) -> np.ndarray:
    """Exact KNN-Shapley values, shape ``(n_train,)``, averaged over every validation point.

    ``standardize=True`` z-scores ``X_train``/``X_val`` on the training columns' mean/std (distance
    sanity: unscaled columns of very different magnitude would dominate the nearest-neighbor ordering).
    Validation points are processed in chunks of ``batch_val`` (each chunk materializes one
    ``(batch_val, n_train)`` distance matrix, bounding peak memory regardless of ``n_val``); chunks run
    in parallel via ``joblib`` when ``n_jobs != 1``.

    Classification only: raises ``NotImplementedError`` for continuous ``y_val`` (the closed form is
    derived for a KNN classification-agreement utility; use ``tmc_shapley`` for regression).
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_val = np.ascontiguousarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if np.issubdtype(y_train.dtype, np.floating) and not np.array_equal(y_train, y_train.astype(np.int64)):
        raise NotImplementedError(
            "knn_shapley: v1 supports classification labels only (the closed form is a KNN "
            "classification-agreement utility) -- continuous y_train detected; use tmc_shapley for regression."
        )

    if standardize:
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma = np.where(sigma == 0.0, 1.0, sigma)
        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    if n_val == 0:
        return np.zeros(n_train, dtype=np.float64)

    batches = [(X_val[start : start + batch_val], y_val[start : start + batch_val]) for start in range(0, n_val, batch_val)]
    if len(batches) == 1 or n_jobs == 1:
        # A joblib Parallel pool's spawn/teardown overhead (~0.8s measured, thread pool included) swamps
        # a single batch's actual compute -- skip the pool entirely rather than pay it for nothing.
        results = [_score_one_batch(Xb, yb, X_train, y_train, k, metric) for Xb, yb in batches]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(_score_one_batch)(Xb, yb, X_train, y_train, k, metric) for Xb, yb in batches)
    values = np.sum(np.asarray(results), axis=0) / n_val
    return np.asarray(values, dtype=np.float64)


__all__ = ["knn_shapley"]
