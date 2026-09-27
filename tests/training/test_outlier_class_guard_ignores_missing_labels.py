"""The outlier-detection guard against wiping out a class ignores missing labels.

``np.unique`` counts NaN as a class, so a binary target left with {0, NaN} after outlier removal still looked like two
classes: the detector could drop every labelled positive without the guard firing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._setup_helpers_outliers import _apply_outlier_detection_global


class _DropRowsDetector:
    """Flags a fixed set of train rows as outliers."""

    def __init__(self, outlier_rows):
        self.outlier_rows = set(outlier_rows)
        self._n_train = 0

    def fit(self, X):
        """Remember how many rows the detector was fitted on."""
        self._n_train = len(X)
        return self

    def predict(self, X):
        """Flag the configured rows as outliers on the fit frame; everything else is an inlier."""
        if len(X) != self._n_train:
            return np.ones(len(X), dtype=int)
        return np.array([-1 if i in self.outlier_rows else 1 for i in range(len(X))])


def test_losing_every_labelled_positive_is_caught_despite_missing_labels():
    """Outlier removal that drops every labelled positive is caught even when unlabelled rows remain."""
    n = 400
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = np.zeros(n)
    y[:3] = 1.0  # the only positives
    y[3:40] = np.nan  # unlabelled rows stay after OD
    filtered_train_df = _apply_outlier_detection_global(
        train_df=train_df, val_df=None, train_idx=np.arange(n), val_idx=None, outlier_detector=_DropRowsDetector(range(3)),
        od_val_set=False, verbose=False, targets_for_classbalance={"hired": y},
    )[0]
    assert len(filtered_train_df) == n, "the OD filter removed the only positives; the class guard should have kept train intact"
