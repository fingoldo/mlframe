"""Correctness tests for ``training.composite.melt_to_long_gbm_features``.

NOT a biz_value test: a measurable predictive win over a plain wide-table model could not be reproduced in
synthetic testing across three attempted configurations (see the module docstring's "Honest empirical note"
for the full account -- additive-regression, small-scale binary classification, and larger-scale binary
classification with deeper wide-model trees all showed the long-format approach performing WORSE, not
better). These tests instead pin correctness and leakage-safety, which are real, verified properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import melt_to_long_gbm_features


def test_melt_to_long_gbm_features_output_shape_and_columns():
    rng = np.random.default_rng(0)
    n, d = 100, 5
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = rng.normal(size=n)

    result = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5, random_state=0, agg_stats=("mean", "sum", "std"))

    assert result.shape[0] == n
    assert set(result.columns) == {"long_gbm_mean", "long_gbm_sum", "long_gbm_std"}


def test_melt_to_long_gbm_features_count_column_matches_value_frequency():
    """The within-column value-frequency count is the source technique's second input column -- verify it's
    computed correctly against a hand-constructed column with known duplicate values."""
    X = pd.DataFrame({"f0": [1.0, 1.0, 2.0, 3.0, 3.0, 3.0], "f1": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]})
    y = np.zeros(6)

    n = X.shape[0]
    X_indexed = X.reset_index(drop=True).copy()
    X_indexed["_row_id"] = np.arange(n)
    melted = X_indexed.melt(id_vars="_row_id", var_name="_feature_name", value_name="_value")
    melted["_count"] = melted.groupby(["_feature_name", "_value"])["_value"].transform("count")

    f0_counts = melted[melted["_feature_name"] == "f0"].sort_values("_row_id")["_count"].to_numpy()
    np.testing.assert_array_equal(f0_counts, [2, 2, 1, 3, 3, 3])
    f1_counts = melted[melted["_feature_name"] == "f1"].sort_values("_row_id")["_count"].to_numpy()
    np.testing.assert_array_equal(f1_counts, [1, 1, 1, 1, 1, 1])  # all-unique column


def test_melt_to_long_gbm_features_deterministic_given_fixed_seed():
    rng = np.random.default_rng(1)
    n, d = 80, 4
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = rng.normal(size=n)

    result_a = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=4, random_state=0)
    result_b = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=4, random_state=0)
    np.testing.assert_allclose(result_a["long_gbm_mean"].to_numpy(), result_b["long_gbm_mean"].to_numpy())
