"""Dealing with outliers in ML pipelines."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from sklearn.ensemble import IsolationForest

# imblearn is optional at module import time — older installations are often
# broken against new sklearn (parse_version moved). `reject_outliers` imports
# it lazily so the rest of the module stays usable.
try:
    from imblearn import FunctionSampler  # noqa: F401
    from imblearn.pipeline import Pipeline as _ImbPipeline
    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    _HAS_IMBLEARN = False
    _ImbPipeline = None  # type: ignore

from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def reject_outliers(
    X: object,
    y: object,
    model: object = None,
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
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # pragma: no cover
        def wrap(fn):
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
    return score


@njit(cache=True)
def count_num_outofranges(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """For each row of X, count how many features fall outside [mins, maxs]."""
    n, d = X.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = 0
        for j in range(d):
            v = X[i, j]
            if v < mins[j] or v > maxs[j]:
                c += 1
        out[i] = c
    return out


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
    mins = np.nanmin(X_train, axis=0)
    maxs = np.nanmax(X_train, axis=0)
    d = X_test.shape[1]
    counts = count_num_outofranges(X_test, mins, maxs)
    return counts.astype(np.float64) / max(d, 1)
