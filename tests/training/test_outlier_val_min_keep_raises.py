"""Regression: val-side outlier detection must RAISE on a collapsed val (mirroring the train-side
min_keep guard), not silently return the unfiltered / 0-row val.

Pre-fix: ``_apply_outlier_detection_global`` logged an error and returned the ORIGINAL (unfiltered,
outlier-contaminated) val_df when OD dropped val below the 1% min_keep floor. A contaminated or 0-row
val silently biases early stopping (val is the ES detector). The train side already raised in the
same situation, so the two sides were inconsistent.

Post-fix: the val side raises a clear ValueError when OD would drop val below min_keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core._setup_helpers_outliers import _apply_outlier_detection_global


class _RejectAllValDetector:
    """Inlier on every train row, outlier on every val row -> val collapses to 0 kept."""

    def __init__(self) -> None:
        self._n_train = 0

    def fit(self, X):  # noqa: N803
        self._n_train = len(X)
        return self

    def predict(self, X):  # noqa: N803
        # Train frame (seen at fit) -> all inliers; anything else (val) -> all outliers.
        if len(X) == self._n_train:
            return np.ones(len(X), dtype=int)
        return -np.ones(len(X), dtype=int)


def _frames(n_train: int = 200, n_val: int = 80):
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x0": rng.normal(size=n_train), "x1": rng.normal(size=n_train)})
    val_df = pd.DataFrame({"x0": rng.normal(size=n_val), "x1": rng.normal(size=n_val)})
    return train_df, val_df


def test_val_side_raises_when_outlier_detection_collapses_val_below_min_keep():
    train_df, val_df = _frames()
    train_idx = np.arange(len(train_df))
    val_idx = np.arange(len(val_df))

    with pytest.raises(ValueError, match=r"val samples.*min_keep guard"):
        _apply_outlier_detection_global(
            train_df=train_df,
            val_df=val_df,
            train_idx=train_idx,
            val_idx=val_idx,
            outlier_detector=_RejectAllValDetector(),
            od_val_set=True,
            verbose=False,
        )
