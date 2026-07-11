"""biz_value test for ``feature_engineering.row_wise_extremality.row_wise_extremality_index``.

Generalizes the row-wise-missing-value-count idea (a compact per-row unusualness signal) to observed values:
when features are on wildly different raw scales, a naive row-wise mean/abs-mean is dominated by whichever
feature happens to have the largest scale, even if that feature carries no signal. Putting every column on
its own within-column rank scale first (this module) should recover the anomaly signal that the naive
scale-sensitive row summary misses entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.row_wise_extremality import row_wise_extremality_index


def _make_mixed_scale_anomaly_data(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    X[:, 0] *= 1000  # huge-scale noise feature -- dominates a naive row mean but carries no signal.

    is_anomaly = rng.random(n) < 0.1
    for j in range(1, n_features):
        X[is_anomaly, j] += rng.choice([-1, 1], is_anomaly.sum()) * 4.0  # signal is on the SMALL-scale columns.

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return df, is_anomaly


def test_biz_val_extremality_index_beats_naive_row_mean_under_mixed_scales():
    df, is_anomaly = _make_mixed_scale_anomaly_data(n=2000, n_features=10, seed=0)

    extremality = row_wise_extremality_index(df)
    naive_row_mean_abs = df.abs().mean(axis=1)

    auc_extremality = roc_auc_score(is_anomaly, extremality)
    auc_naive = roc_auc_score(is_anomaly, naive_row_mean_abs)

    assert auc_extremality >= 0.95, f"expected the extremality index to cleanly separate anomalous rows, got auc={auc_extremality:.4f}"
    assert auc_extremality > auc_naive + 0.3, f"expected the extremality index to beat the naive scale-sensitive row mean by a wide margin, got extremality={auc_extremality:.4f} naive={auc_naive:.4f}"


def test_row_wise_extremality_index_symmetric_around_median():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [100, 102, 104, 106, 108]})
    result = row_wise_extremality_index(df)

    assert result.iloc[2] == 0.0  # the exact median row of both columns.
    assert result.iloc[0] == result.iloc[4]  # symmetric: equidistant extremes score equally.
    assert result.iloc[0] > result.iloc[1] > result.iloc[2]  # monotonically increasing toward the extremes.


def test_row_wise_extremality_index_handles_nan():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = row_wise_extremality_index(df)
    assert not result.isna().any()  # every row still gets a value from its non-NaN columns.
