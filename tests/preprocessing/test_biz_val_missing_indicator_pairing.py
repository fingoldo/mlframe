"""biz_value test for ``preprocessing.missing_indicator_pairing.impute_with_missing_indicator``.

Synthetic: an MNAR (missing-not-at-random) feature where the FACT of missingness is predictive of the target,
independent of the imputed value itself. Plain median-imputation-only discards that signal (every imputed row
gets the identical fill value, indistinguishable from a genuinely-observed value near the median); the paired
"was_missing" indicator recovers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.preprocessing.missing_indicator_pairing import impute_with_missing_indicator


def _make_mnar_dataset(n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_rows)
    is_missing = rng.random(n_rows) < 0.4
    # x itself carries only WEAK signal; missingness is the DOMINANT predictor -- an imputation-only model
    # (whose sole feature is x, identical fill value for every missing row) can only exploit the weak x signal
    # and is blind to the strong missingness signal, while the paired indicator directly exposes it.
    p = np.where(is_missing, 0.85, 0.5 + 0.05 * np.sign(x))
    y = (rng.random(n_rows) < p).astype(np.int64)
    x_observed = x.copy()
    x_observed[is_missing] = np.nan
    return pd.DataFrame({"x": x_observed}), y


def test_biz_val_missing_indicator_pairing_recovers_mnar_signal():
    df, y = _make_mnar_dataset(n_rows=4000, seed=0)
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)

    median_fill = df_train["x"].median()
    X_train_impute_only = df_train[["x"]].fillna(median_fill)
    X_test_impute_only = df_test[["x"]].fillna(median_fill)
    model_impute_only = LogisticRegression().fit(X_train_impute_only, y_train)
    auc_impute_only = roc_auc_score(y_test, model_impute_only.predict_proba(X_test_impute_only)[:, 1])

    df_train_paired = impute_with_missing_indicator(df_train, columns=["x"])
    df_test_paired = impute_with_missing_indicator(df_test, columns=["x"])
    assert "x_was_missing" in df_train_paired.columns
    assert df_train_paired["x"].isna().sum() == 0

    feature_cols = ["x", "x_was_missing"]
    model_paired = LogisticRegression().fit(df_train_paired[feature_cols], y_train)
    auc_paired = roc_auc_score(y_test, model_paired.predict_proba(df_test_paired[feature_cols])[:, 1])

    assert auc_paired >= auc_impute_only + 0.05, f"expected the paired indicator to recover MNAR signal, got paired={auc_paired:.4f} impute_only={auc_impute_only:.4f}"


def test_missing_indicator_pairing_leaves_complete_columns_untouched():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [np.nan, 1.0, 2.0]})
    out = impute_with_missing_indicator(df)
    assert "x_was_missing" not in out.columns
    assert "y_was_missing" in out.columns
    assert out["y"].isna().sum() == 0


def test_missing_indicator_pairing_respects_explicit_fill_values():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    out = impute_with_missing_indicator(df, fill_values={"x": -999.0})
    assert out.loc[1, "x"] == -999.0
    assert out.loc[1, "x_was_missing"] == True  # noqa: E712


def test_missing_indicator_pairing_mode_strategy():
    df = pd.DataFrame({"c": ["a", "a", "b", None]})
    out = impute_with_missing_indicator(df, strategy="mode")
    assert out.loc[3, "c"] == "a"
    assert out.loc[3, "c_was_missing"] == True  # noqa: E712


def test_missing_indicator_pairing_rejects_unknown_strategy():
    import pytest

    df = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ValueError):
        impute_with_missing_indicator(df, strategy="bogus")
