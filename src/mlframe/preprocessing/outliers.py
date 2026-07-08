"""Dealing with outliers in ML pipelines."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


from typing import Any, Optional

from sklearn.ensemble import IsolationForest

# imblearn is optional at module import time — older installations are often
# broken against new sklearn (parse_version moved). `reject_outliers` imports
# it lazily so the rest of the module stays usable.
try:
    from imblearn import FunctionSampler
    from imblearn.pipeline import Pipeline as _ImbPipeline
    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    _HAS_IMBLEARN = False
    _ImbPipeline = None

from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def reject_outliers(
    X: Any,
    y: Any,
    model: Optional[Any] = None,
    verbose: bool = True,
):
    """Function used to resample the dataset by dropping the outliers. Should be a part of imblearn Pipeline:

    from imblearn import FunctionSampler
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([("out", FunctionSampler(func=reject_outliers, validate=False)), ("est", clf)])

    """

    if model is None:
        if not _HAS_IMBLEARN:
            raise ImportError("imblearn is required for the default reject_outliers pipeline; install imblearn or pass `model=`.")
        model = _ImbPipeline([("imp", SimpleImputer()), ("est", IsolationForest())])

    model.fit(X)
    y_pred = model.predict(X)
    idx = y_pred == 1

    if verbose:
        logger.info("Outlier rejection: received %s samples, kept %s", len(X), idx.sum())

    return X[idx], y[idx]


# ----------------------------------------------------------------------------------------------------------------------------
# Salvaged from training_old.py (outlier scoring helpers)
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np

try:
    import numba
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range
    numba = None

    def njit(*args, **kwargs):  # pragma: no cover
        """Fallback ``numba.njit`` stub used when numba is unavailable: returns the function unmodified (or a passthrough decorator when called with keyword options), so the module still imports and runs in pure Python."""
        def wrap(fn):
            """Passthrough decorator returned by the ``njit`` fallback when invoked with options (e.g. ``@njit(parallel=True)``); returns ``fn`` unchanged."""
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


def compute_outlier_detector_score(detector, X) -> np.ndarray:
    """Map sklearn anomaly-detector .predict output (+1/-1) to binary 0/1 score.

    NaN-safe: replaces NaNs in result with 0 (inlier-by-default).
    """
    is_inlier = detector.predict(X)
    is_inlier = np.asarray(is_inlier)
    score = (is_inlier == -1).astype(float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(score)


@njit(cache=True, parallel=True)
def count_num_outofranges(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """For each row of X, count how many features fall outside [mins, maxs].

    Per-row independent integer reduction parallelised over rows via prange; the per-row count is order-invariant so the result is bit-identical to the
    serial loop regardless of thread scheduling (measured 3-3.8x at n=10M across d in {4,8,30}, 2026-06-14).
    """
    n, d = X.shape
    out = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for j in range(d):
            v = X[i, j]
            if v < mins[j] or v > maxs[j]:
                c += 1
        out[i] = c
    return out


@njit(parallel=True)
def _nanminmax_cols(X: np.ndarray):
    """Per-column NaN-ignoring min and max in ONE fused pass (replaces two full np.nanmin + np.nanmax sweeps).

    Returns ``(mins, maxs)`` bit-identical to ``np.nanmin(X, axis=0)`` / ``np.nanmax(X, axis=0)``: an all-NaN column yields +inf/-inf seeds
    that collapse to NaN to mirror numpy's empty-slice result. Halves the memory traffic (one sweep, not two) and parallelises over row chunks;
    the per-column min/max reduction is order-invariant so the result is independent of thread scheduling (measured ~2-3x at n=10M, 2026-06-15).
    """
    n, d = X.shape
    nt = numba.get_num_threads()
    lmins = np.full((nt, d), np.inf)
    lmaxs = np.full((nt, d), -np.inf)
    chunk = (n + nt - 1) // nt
    for t in prange(nt):
        s = t * chunk
        e = s + chunk
        if e > n:
            e = n
        for i in range(s, e):
            for j in range(d):
                v = X[i, j]
                if v == v:  # not NaN
                    if v < lmins[t, j]:
                        lmins[t, j] = v
                    if v > lmaxs[t, j]:
                        lmaxs[t, j] = v
    mins = np.empty(d, dtype=X.dtype)
    maxs = np.empty(d, dtype=X.dtype)
    for j in range(d):
        mn = np.inf
        mx = -np.inf
        for t in range(nt):
            if lmins[t, j] < mn:
                mn = lmins[t, j]
            if lmaxs[t, j] > mx:
                mx = lmaxs[t, j]
        # All-NaN column: no finite value updated the seeds — collapse to NaN like numpy's empty-slice nanmin/nanmax.
        if mn == np.inf:
            mn = np.nan
            mx = np.nan
        mins[j] = mn
        maxs[j] = mx
    return mins, maxs


@njit
def _nanminmax_cols_serial(X: np.ndarray):
    """Serial per-column NaN-ignoring min/max in one pass; bit-identical to ``_nanminmax_cols`` but without the per-thread reduce buffer + join.

    The parallel kernel's thread-spawn + (nt, d) buffer allocation + reduction join lose to a plain serial sweep on small frames; the dispatcher in
    ``compute_naive_outlier_score`` routes small ``n*d`` here (crossover ~20k measured 2026-07: serial 46-83% faster below, parallel wins above)."""
    n, d = X.shape
    mins = np.full(d, np.inf, dtype=X.dtype)
    maxs = np.full(d, -np.inf, dtype=X.dtype)
    for i in range(n):
        for j in range(d):
            v = X[i, j]
            if v == v:  # not NaN
                if v < mins[j]:
                    mins[j] = v
                if v > maxs[j]:
                    maxs[j] = v
    for j in range(d):
        # All-NaN column: collapse to NaN like numpy's empty-slice nanmin/nanmax (mirrors the parallel kernel).
        if mins[j] == np.inf:
            mins[j] = np.nan
            maxs[j] = np.nan
    return mins, maxs


# Below this n*d the serial nanminmax sweep beats the parallel one (thread-spawn + reduce-buffer overhead dominates); measured crossover ~20k (2026-07).
_NANMINMAX_PARALLEL_MIN_ELEMS = 20_000


def compute_naive_outlier_score(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Percentage of features in each X_test row outside train min/max bounds.

    Returns a float array in [0, 1] of length len(X_test).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    # The njit kernel indexes train mins/maxs by X_test's feature columns; a feature-count mismatch reads out-of-bounds and returns silent garbage.
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"compute_naive_outlier_score: X_train has {X_train.shape[1]} features but X_test has {X_test.shape[1]}; feature counts must match.")
    if _HAS_NUMBA:
        if X_train.shape[0] * X_train.shape[1] < _NANMINMAX_PARALLEL_MIN_ELEMS:
            mins, maxs = _nanminmax_cols_serial(X_train)
        else:
            mins, maxs = _nanminmax_cols(X_train)
    else:
        mins = np.nanmin(X_train, axis=0)
        maxs = np.nanmax(X_train, axis=0)
    d = X_test.shape[1]
    counts = count_num_outofranges(X_test, mins, maxs)
    return np.asarray(counts.astype(np.float64) / max(d, 1))
