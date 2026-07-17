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

    assert auc_paired >= auc_impute_only + 0.05, (
        f"expected the paired indicator to recover MNAR signal, got paired={auc_paired:.4f} impute_only={auc_impute_only:.4f}"
    )


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


def _make_grouped_income_dataset(n_rows: int, seed: int):
    """Synthetic "income" whose true group-conditional median differs sharply by "region" -- region A's
    typical income is ~30k, region B's is ~120k. Missingness is independent of region (MCAR within group),
    so a single global median (blended across both regions, ~75k) is a poor fill for either region, while
    the group-conditional median recovers each region's own typical value closely.
    """
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, size=n_rows)  # 0 -> "A" (low income), 1 -> "B" (high income)
    true_income = np.where(region == 0, rng.normal(30_000, 4_000, size=n_rows), rng.normal(120_000, 4_000, size=n_rows))
    is_missing = rng.random(n_rows) < 0.4
    observed_income = true_income.copy()
    observed_income[is_missing] = np.nan
    df = pd.DataFrame({"income": observed_income, "region": np.where(region == 0, "A", "B")})
    return df, true_income, is_missing


def test_biz_val_missing_indicator_pairing_group_conditional_lowers_imputation_rmse():
    df, true_income, is_missing = _make_grouped_income_dataset(n_rows=4000, seed=0)

    out_global = impute_with_missing_indicator(df, columns=["income"])
    out_grouped = impute_with_missing_indicator(df, columns=["income"], group_col="region")

    rmse_global = np.sqrt(np.mean((out_global.loc[is_missing, "income"].to_numpy() - true_income[is_missing]) ** 2))
    rmse_grouped = np.sqrt(np.mean((out_grouped.loc[is_missing, "income"].to_numpy() - true_income[is_missing]) ** 2))

    assert rmse_grouped <= 0.35 * rmse_global, (
        f"expected group-conditional imputation to cut RMSE well below global, got grouped={rmse_grouped:.1f} global={rmse_global:.1f}"
    )


def test_biz_val_missing_indicator_pairing_group_conditional_improves_downstream_model():
    df, true_income, is_missing = _make_grouped_income_dataset(n_rows=4000, seed=1)
    y = (true_income > 75_000).astype(np.int64)  # downstream target correlated with the TRUE income
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)

    out_train_global = impute_with_missing_indicator(df_train, columns=["income"])
    out_test_global = impute_with_missing_indicator(df_test, columns=["income"])
    model_global = LogisticRegression().fit(out_train_global[["income"]], y_train)
    auc_global = roc_auc_score(y_test, model_global.predict_proba(out_test_global[["income"]])[:, 1])

    out_train_grouped = impute_with_missing_indicator(df_train, columns=["income"], group_col="region")
    out_test_grouped = impute_with_missing_indicator(df_test, columns=["income"], group_col="region")
    model_grouped = LogisticRegression().fit(out_train_grouped[["income"]], y_train)
    auc_grouped = roc_auc_score(y_test, model_grouped.predict_proba(out_test_grouped[["income"]])[:, 1])

    assert auc_grouped >= auc_global + 0.08, (
        f"expected group-conditional imputation to materially improve downstream AUC, got grouped={auc_grouped:.4f} global={auc_global:.4f}"
    )


def test_missing_indicator_pairing_group_col_none_matches_previous_global_behavior():
    """Regression: group_col defaults to None and must reproduce the pre-extension global-only behavior bit-for-bit."""
    df, _, _ = _make_grouped_income_dataset(n_rows=500, seed=2)
    out_default = impute_with_missing_indicator(df, columns=["income"])

    median_fill = df["income"].median()
    expected = df["income"].fillna(median_fill)

    pd.testing.assert_series_equal(out_default["income"], expected, check_names=False)


def test_missing_indicator_pairing_group_col_fills_empty_group_with_global_fallback():
    """A group with zero non-missing values must fall back to the global statistic, never leave a NaN fill."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, np.nan, np.nan],
            "grp": ["A", "A", "B", "B", "C"],  # group "C" has no non-missing "x" at all
        }
    )
    out = impute_with_missing_indicator(df, columns=["x"], group_col="grp")
    assert out["x"].isna().sum() == 0
    # group "C" falls back to the global median of the observed values (1.0, 2.0, 3.0) = 2.0
    assert out.loc[4, "x"] == 2.0
    # group "B" has one observed value (3.0); its own-group fill should be used, not the global median
    assert out.loc[3, "x"] == 3.0
