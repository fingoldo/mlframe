"""Sensor for the centralised synthetic-data builders in ``tests/training/synthetic.py``.

Pins shape contracts + determinism so that future edits to the builders cannot silently change the per-call return
signature seen by the ~10-15 migrated test sites.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.training.synthetic import (
    make_categorical_classification_data,
    make_informative_noise_classification,
    make_outlier_regression_data,
    make_simple_classification_data,
    make_simple_regression_data,
    make_sklearn_classification_df,
    make_sklearn_regression_df,
)


def test_make_simple_classification_data_shape_and_determinism():
    df1, names1, cats1, y1 = make_simple_classification_data(n_samples=200, n_features=8, seed=7)
    df2, names2, cats2, y2 = make_simple_classification_data(n_samples=200, n_features=8, seed=7)
    assert df1.shape == (200, 8 + 1)
    assert "target" in df1.columns
    assert len(names1) == 8
    assert cats1 == []
    assert np.array_equal(y1, y2)
    pd.testing.assert_frame_equal(df1, df2)


def test_make_simple_regression_data_shape_and_determinism():
    df1, names1, y1 = make_simple_regression_data(n_samples=150, n_features=6, seed=11)
    df2, _, y2 = make_simple_regression_data(n_samples=150, n_features=6, seed=11)
    assert df1.shape == (150, 7)
    assert len(names1) == 6
    assert np.allclose(y1, y2)


def test_make_categorical_classification_data_has_cat_features():
    df, names, cats, y = make_categorical_classification_data(n_samples=300, n_numeric=4, seed=3)
    assert df.shape == (300, 4 + 3 + 1)
    assert set(cats).issubset(set(df.columns))
    assert all(df[c].dtype == object for c in cats)
    assert len(y) == 300


def test_make_outlier_regression_data_has_outliers():
    df, names, y = make_outlier_regression_data(n_samples=400, n_features=5, outlier_fraction=0.1, seed=0)
    assert df.shape == (400, 6)
    # 10% of rows should have |residual| much larger than the clean tail
    abs_y = np.abs(y - np.median(y))
    n_big = int((abs_y > 5.0).sum())
    assert n_big >= 30, f"expected ~40 outliers, got {n_big}"


def test_make_sklearn_classification_df_shape():
    X_df, y, names = make_sklearn_classification_df(n_samples=100, n_features=12, n_informative=3, seed=5)
    assert X_df.shape == (100, 12)
    assert names == [f"f{i}" for i in range(12)]
    assert set(np.unique(y)) <= {0, 1}


def test_make_sklearn_classification_df_column_prefix():
    X_df, _, names = make_sklearn_classification_df(n_samples=20, n_features=4, n_informative=2, column_prefix="feature_", seed=0)
    assert names == ["feature_0", "feature_1", "feature_2", "feature_3"]


def test_make_sklearn_regression_df_shape():
    X_df, y, names = make_sklearn_regression_df(n_samples=50, n_features=6, n_informative=3, seed=0)
    assert X_df.shape == (50, 6)
    assert y.shape == (50,)


def test_make_informative_noise_classification_shape_and_balance():
    X_df, y, informative_idx = make_informative_noise_classification(n_samples=400, n_informative=4, n_noise=6, seed=0)
    assert X_df.shape == (400, 10)
    assert informative_idx == [0, 1, 2, 3]
    assert {int(c) for c in np.unique(y)} <= {0, 1}
    # Sign-of-linear-combination label should give roughly 50/50; allow wide window because we use 3 of 4 informatives.
    frac_pos = float(y.mean())
    assert 0.30 <= frac_pos <= 0.70


def test_make_informative_noise_classification_determinism():
    a = make_informative_noise_classification(n_samples=100, seed=99)
    b = make_informative_noise_classification(n_samples=100, seed=99)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
