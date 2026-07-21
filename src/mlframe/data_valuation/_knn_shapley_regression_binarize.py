"""Regression row-valuation via median-split binarization + exact KNN-Shapley.

``knn_shapley``'s closed-form recursion is derived for a KNN classification-agreement utility (see its
own module docstring) -- there is no analogous closed form for a continuous target. This module tries
the obvious cheap workaround instead of reaching straight for the expensive model-agnostic
``tmc_shapley``: binarize the continuous target by a TRAIN-derived split point (median by default,
i.e. "top half" vs "bottom half"), then run the exact classification closed form on the binarized
labels as a proxy for the continuous target's row valuation.

HONEST SCOPE: this is a proxy, not an exact regression Shapley value -- a row whose true y value is
wildly wrong but happens to land on the correct SIDE of the split (e.g. both the true and corrupted
value are above the median) will not be flagged. Whether this proxy correlates well enough with
genuine row usefulness to be practically useful is an empirical question, answered by this package's
biz_val test (see ``tests/data_valuation/test_biz_val_knn_shapley_regression_binarize.py``) -- read
that test's docstring for the measured verdict before relying on this for a real regression dataset.
"""

from __future__ import annotations

import numpy as np

from mlframe.data_valuation._knn_shapley import knn_shapley


def knn_shapley_regression_binarize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    split: str = "median",
    k: int = 5,
    metric: str = "euclidean",
    standardize: bool = True,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict]:
    """KNN-Shapley row values for a CONTINUOUS target, via median/quantile binarization.

    The split point is computed from ``y_train`` ONLY (never ``y_val``) and applied to both arrays,
    so ``y_val``'s own distribution can never leak into the threshold choice. ``split="median"``
    (default) splits at the train median (~50/50 top/bottom half); a float in ``(0, 1)`` uses that
    quantile instead (e.g. ``split=0.75`` labels the top quartile as the positive class).

    Returns ``(values, info)`` where ``info`` holds ``threshold`` (the actual train-derived cutoff
    value) and ``train_positive_frac`` (diagnostic: how balanced the binarized labels ended up,
    useful for spotting a degenerate near-constant target).
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)

    if split == "median":
        quantile = 0.5
    else:
        quantile = float(split)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"knn_shapley_regression_binarize: split must be 'median' or a float in (0, 1); got {split!r}")

    threshold = float(np.quantile(y_train, quantile))
    y_train_bin = (y_train > threshold).astype(np.int64)
    y_val_bin = (y_val > threshold).astype(np.int64)

    values = knn_shapley(X_train, y_train_bin, X_val, y_val_bin, k=k, metric=metric, standardize=standardize, n_jobs=n_jobs)
    info = dict(threshold=threshold, train_positive_frac=float(y_train_bin.mean()))
    return values, info


__all__ = ["knn_shapley_regression_binarize"]
