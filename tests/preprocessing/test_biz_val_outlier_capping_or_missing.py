"""biz_value test for ``preprocessing.outlier_capping_or_missing.outlier_cap_or_missing``.

Synthetic: a feature with a few extreme corrupted values (data-entry-error-style, magnitude unrelated to the
true relationship) injected into an otherwise clean linear signal. Both ``cap`` and ``missing_impute`` modes
should recover a downstream Ridge model's RMSE close to the outlier-free baseline, both far better than leaving
the extreme values untreated (which drag a squared-error fit badly off).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.preprocessing.outlier_capping_or_missing import outlier_cap_or_missing


def _make_dataset_with_outliers(n_rows: int, seed: int, outlier_frac: float = 0.02):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_rows)
    y = 2.0 * x + rng.normal(scale=0.3, size=n_rows)
    x_corrupted = x.copy()
    n_outliers = int(n_rows * outlier_frac)
    outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    x_corrupted[outlier_idx] = rng.uniform(50, 100, size=n_outliers) * rng.choice([-1, 1], size=n_outliers)
    return pd.DataFrame({"x": x_corrupted}), y


def _fit_rmse(X_train, y_train, X_test, y_test) -> float:
    model = Ridge().fit(X_train, y_train)
    return float(mean_squared_error(y_test, model.predict(X_test)) ** 0.5)


def test_biz_val_outlier_cap_or_missing_beats_untreated_outliers():
    df, y = _make_dataset_with_outliers(n_rows=3000, seed=0)
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    rmse_untreated = _fit_rmse(df_train[["x"]], y_train, df_test[["x"]], y_test)

    df_train_cap = outlier_cap_or_missing(df_train, mode="cap")
    df_test_cap = outlier_cap_or_missing(df_test, mode="cap")
    rmse_cap = _fit_rmse(df_train_cap[["x"]], y_train, df_test_cap[["x"]], y_test)

    df_train_missing = outlier_cap_or_missing(df_train, mode="missing_impute")
    df_test_missing = outlier_cap_or_missing(df_test, mode="missing_impute")
    rmse_missing = _fit_rmse(df_train_missing[["x"]], y_train, df_test_missing[["x"]], y_test)

    assert rmse_cap < rmse_untreated * 0.5, f"expected capping to cut RMSE by >=50% vs untreated outliers, got cap={rmse_cap:.4f} untreated={rmse_untreated:.4f}"
    assert rmse_missing < rmse_untreated * 0.5, f"expected missing-impute to cut RMSE by >=50% vs untreated outliers, got missing={rmse_missing:.4f} untreated={rmse_untreated:.4f}"


def test_outlier_cap_or_missing_cap_mode_clips_within_bounds():
    df, _ = _make_dataset_with_outliers(n_rows=2000, seed=1)
    out = outlier_cap_or_missing(df, mode="cap")
    # After capping, no value should be as extreme as the injected corruption range.
    assert out["x"].abs().max() < 40.0


def test_outlier_cap_or_missing_missing_impute_mode_has_no_extreme_values():
    df, _ = _make_dataset_with_outliers(n_rows=2000, seed=2)
    out = outlier_cap_or_missing(df, mode="missing_impute")
    assert out["x"].isna().sum() == 0
    assert out["x"].abs().max() < 40.0


def test_outlier_cap_or_missing_rejects_unknown_mode():
    import pytest

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        outlier_cap_or_missing(df, mode="bogus")


def test_outlier_cap_or_missing_skewness_driven_rule_selection():
    rng = np.random.default_rng(3)
    symmetric = rng.normal(size=2000)
    skewed = rng.exponential(size=2000)
    df = pd.DataFrame({"symmetric": symmetric, "skewed": skewed})
    # Should not raise, and should apply different bound rules internally (IQR for skewed, mean+/-3std for
    # symmetric) without erroring on either.
    out = outlier_cap_or_missing(df, mode="cap")
    assert out.shape == df.shape
