"""biz_value test for ``feature_engineering.sentinel_missing_count.add_sentinel_missing_count_feature``.

The win (2nd_porto-seguro-safe-driver-prediction.md): many real datasets encode missingness as an explicit
SENTINEL value (-1, -999, ...) rather than true NaN. mlframe's existing NaN-based row-missing-count generator
(``feature_selection.filters._missingness_fe``) is blind to this -- it would report ZERO missing values on a
sentinel-encoded frame, even when the sentinel count is itself a genuinely predictive signal (e.g. rows with
more missing fields correlate with the target, a classic insurance/credit-risk pattern). This test confirms
the sentinel-count feature recovers signal a naive NaN-based (or no) missing-count feature would miss
entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.sentinel_missing_count import add_sentinel_missing_count_feature


def _make_sentinel_encoded_dataset(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # missing_rate driven by y itself (rows with the target condition tend to have more sentinel-encoded
    # fields) -- a realistic "missingness is informative" pattern, matching the source's own finding.
    y = rng.integers(0, 2, n)
    missing_rate_per_row = 0.05 + 0.35 * y  # y=1 rows have far more sentinel fields
    mask = rng.random((n, n_features)) < missing_rate_per_row[:, None]
    X_sentinel = np.where(mask, -1.0, X)  # sentinel value -1, NOT NaN

    df = pd.DataFrame(X_sentinel, columns=[f"f{i}" for i in range(n_features)])
    return df, y


def test_biz_val_sentinel_missing_count_recovers_signal_naive_nan_count_misses():
    df, y = _make_sentinel_encoded_dataset(n=2000, n_features=15, seed=0)

    # A naive NaN-based missing-count is blind to sentinel-encoded missingness -- the column is ALWAYS 0.
    naive_nan_count = df.isna().sum(axis=1).to_numpy()
    assert (naive_nan_count == 0).all(), "sanity check: the synthetic sentinel-encoded frame has no true NaNs"

    df_with_sentinel_count = add_sentinel_missing_count_feature(df, sentinel=-1.0)
    sentinel_count_only = df_with_sentinel_count[["sentinel_missing_count"]].to_numpy()

    auc_sentinel_count_alone = cross_val_score(LogisticRegression(max_iter=500), sentinel_count_only, y, cv=5, scoring="roc_auc").mean()
    auc_naive_nan_count_alone = 0.5  # a constant all-zero feature carries zero information -> exactly chance

    assert auc_sentinel_count_alone > 0.75, f"expected the sentinel-count feature alone to carry strong signal, got AUC={auc_sentinel_count_alone:.4f}"
    assert auc_sentinel_count_alone > auc_naive_nan_count_alone + 0.2


def test_add_sentinel_missing_count_feature_exact_counts():
    df = pd.DataFrame({"a": [-1, 0, -1], "b": [-1, -1, 5], "c": [3, -1, -1]})
    out = add_sentinel_missing_count_feature(df, sentinel=-1)
    np.testing.assert_array_equal(out["sentinel_missing_count"], [2, 2, 2])


def test_add_sentinel_missing_count_feature_respects_column_subset():
    df = pd.DataFrame({"a": [-1, -1], "b": [-1, 5]})
    out = add_sentinel_missing_count_feature(df, sentinel=-1, columns=["a"])
    np.testing.assert_array_equal(out["sentinel_missing_count"], [1, 1])
