"""KNN-Shapley row valuation for mlframe's multi-COLUMN target types (see ``docs/MULTI_OUTPUT.md`` /
``docs/MULTI_TARGET_REGRESSION.md`` for the target-type matrix this mirrors).

``knn_shapley`` itself already covers ``BINARY_CLASSIFICATION`` and ``MULTICLASS_CLASSIFICATION``
unchanged -- the closed-form recursion is a same-class agreement indicator, which generalizes to any
number of classes with no code change (verified empirically, not just assumed). ``REGRESSION`` goes
through :func:`_knn_shapley_regression_binarize.knn_shapley_regression_binarize`.

This module adds the two remaining ``(N, K)``-shaped target types by decomposing them into ``K``
independent single-column valuations (each column IS one of the two already-solved cases) and
averaging the per-row values across columns:
    ``MULTILABEL_CLASSIFICATION`` -- K independent binary columns -> K :func:`knn_shapley` calls.
    ``MULTI_TARGET_REGRESSION`` -- K independent continuous columns -> K
    :func:`knn_shapley_regression_binarize` calls.

A row's averaged value answers "how useful is this row's (X, Y-row) pair across the WHOLE target
vector" -- a row that is informative for target 3 but harmful for target 1 nets out near its true
mixed contribution, which is the natural cooperative-game reading when a single shared model trunk
(mlframe's actual multi-target training strategy, per the docs above) is fit against all K columns at
once, not a persuasive-sounding rationalization: the biz_val test for each target type measures this
directly rather than asserting it.
"""

from __future__ import annotations

import numpy as np

from mlframe.data_valuation._knn_shapley import knn_shapley
from mlframe.data_valuation._knn_shapley_regression_binarize import knn_shapley_regression_binarize


def knn_shapley_multilabel(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
    standardize: bool = True,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict]:
    """KNN-Shapley row values for ``MULTILABEL_CLASSIFICATION`` (``Y`` shape ``(N, K)`` int binary).

    Runs :func:`knn_shapley` independently per label column and averages the ``K`` per-row value
    vectors. Returns ``(values, info)`` with ``info["per_label_values"]`` (``(K, n_train)``, the raw
    unaveraged per-column values -- useful for spotting a row that is informative for one label and
    harmful for another, which the averaged value alone would mask).
    """
    Y_train = np.asarray(Y_train)
    Y_val = np.asarray(Y_val)
    if Y_train.ndim != 2 or Y_val.ndim != 2:
        raise ValueError(f"knn_shapley_multilabel: Y_train/Y_val must be 2-D (N, K); got shapes {Y_train.shape}, {Y_val.shape}")
    k_labels = Y_train.shape[1]
    if Y_val.shape[1] != k_labels:
        raise ValueError(f"knn_shapley_multilabel: Y_train has {k_labels} label columns, Y_val has {Y_val.shape[1]}")

    per_label_values = np.empty((k_labels, X_train.shape[0]), dtype=np.float64)
    for col in range(k_labels):
        per_label_values[col] = knn_shapley(X_train, Y_train[:, col], X_val, Y_val[:, col], k=k, metric=metric, standardize=standardize, n_jobs=n_jobs)

    values = per_label_values.mean(axis=0)
    info = dict(per_label_values=per_label_values, n_labels=k_labels)
    return values, info


def knn_shapley_multi_target_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    split: str = "median",
    k: int = 5,
    metric: str = "euclidean",
    standardize: bool = True,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict]:
    """KNN-Shapley row values for ``MULTI_TARGET_REGRESSION`` (``Y`` shape ``(N, K)`` float).

    Runs :func:`knn_shapley_regression_binarize` independently per target column (each column gets its
    OWN train-derived median/quantile split -- never a single split across the concatenated columns)
    and averages the ``K`` per-row value vectors. Returns ``(values, info)`` with
    ``info["per_target_values"]`` (``(K, n_train)``, the raw unaveraged per-column values) and
    ``info["thresholds"]`` (the ``K`` per-column split points, for diagnostics).
    """
    Y_train = np.asarray(Y_train, dtype=np.float64)
    Y_val = np.asarray(Y_val, dtype=np.float64)
    if Y_train.ndim != 2 or Y_val.ndim != 2:
        raise ValueError(f"knn_shapley_multi_target_regression: Y_train/Y_val must be 2-D (N, K); got shapes {Y_train.shape}, {Y_val.shape}")
    k_targets = Y_train.shape[1]
    if Y_val.shape[1] != k_targets:
        raise ValueError(f"knn_shapley_multi_target_regression: Y_train has {k_targets} target columns, Y_val has {Y_val.shape[1]}")

    per_target_values = np.empty((k_targets, X_train.shape[0]), dtype=np.float64)
    thresholds = np.empty(k_targets, dtype=np.float64)
    for col in range(k_targets):
        col_values, col_info = knn_shapley_regression_binarize(
            X_train, Y_train[:, col], X_val, Y_val[:, col], split=split, k=k, metric=metric, standardize=standardize, n_jobs=n_jobs
        )
        per_target_values[col] = col_values
        thresholds[col] = col_info["threshold"]

    values = per_target_values.mean(axis=0)
    info = dict(per_target_values=per_target_values, thresholds=thresholds, n_targets=k_targets)
    return values, info


__all__ = ["knn_shapley_multilabel", "knn_shapley_multi_target_regression"]
