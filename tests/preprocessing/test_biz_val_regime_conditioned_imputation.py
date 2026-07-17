"""biz_value test for ``preprocessing.regime_conditioned_median_fill``.

The win: when a feature's distribution genuinely shifts across a regime column (a Jane-Street-market-
prediction writeup's single most valuable trick), filling NaNs with the GLOBAL median imputes badly-wrong
values for whichever regime is far from the population center, while filling with the REGIME-conditioned
median recovers values much closer to the true (masked) ones -- measurably improving downstream prediction
RMSE built on the imputed feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.preprocessing.regime_conditioned_imputation import regime_conditioned_median_fill


def test_biz_val_regime_conditioned_fill_beats_global_median_fill():
    """Regime conditioned fill beats global median fill."""
    rng = np.random.default_rng(0)
    n = 4000

    regime = rng.integers(0, 2, n)
    true_feature = np.where(regime == 0, rng.normal(-5, 1, n), rng.normal(5, 1, n))
    y = 2.0 * true_feature + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"regime": regime, "x": true_feature, "y": y})
    nan_mask = rng.random(n) < 0.3
    df_missing = df.copy()
    df_missing.loc[nan_mask, "x"] = np.nan

    global_filled = df_missing.copy()
    global_filled["x"] = global_filled["x"].fillna(global_filled["x"].median())

    regime_filled = regime_conditioned_median_fill(df_missing, regime_col="regime", feature_cols=["x"])

    # imputation error on the masked rows, against the true (unmasked) value.
    global_impute_error = float(np.mean(np.abs(global_filled.loc[nan_mask, "x"] - true_feature[nan_mask])))
    regime_impute_error = float(np.mean(np.abs(regime_filled.loc[nan_mask, "x"] - true_feature[nan_mask])))
    assert regime_impute_error < global_impute_error * 0.3, (
        f"regime-conditioned fill should recover values far closer to the true masked feature: "
        f"regime={regime_impute_error:.4f} global={global_impute_error:.4f}"
    )

    split = int(0.7 * n)
    model_global = LinearRegression().fit(global_filled[["x"]].iloc[:split], y[:split])
    model_regime = LinearRegression().fit(regime_filled[["x"]].iloc[:split], y[:split])

    rmse_global = float(np.sqrt(mean_squared_error(y[split:], model_global.predict(global_filled[["x"]].iloc[split:]))))
    rmse_regime = float(np.sqrt(mean_squared_error(y[split:], model_regime.predict(regime_filled[["x"]].iloc[split:]))))
    assert rmse_regime < rmse_global, f"regime-conditioned fill should improve downstream RMSE: regime={rmse_regime:.4f} global={rmse_global:.4f}"


def test_regime_conditioned_fill_falls_back_to_global_median_for_all_nan_regime():
    """Regime conditioned fill falls back to global median for all nan regime."""
    df = pd.DataFrame({"regime": [0, 0, 1, 1], "x": [10.0, np.nan, np.nan, np.nan]})
    result = regime_conditioned_median_fill(df, regime_col="regime", feature_cols=["x"])
    assert not result["x"].isna().any()
    assert result["x"].iloc[2] == 10.0  # regime 1 has no observed x -> falls back to global median (10.0)


def test_regime_conditioned_fill_missing_regime_value_falls_back_to_global_median():
    """Regime conditioned fill missing regime value falls back to global median."""
    df = pd.DataFrame({"regime": [0, 0, np.nan], "x": [10.0, 20.0, np.nan]})
    result = regime_conditioned_median_fill(df, regime_col="regime", feature_cols=["x"])
    assert result["x"].iloc[2] == 15.0  # no regime to condition on -> global median


def test_biz_val_regime_conditioned_fill_hierarchical_composite_beats_single_column():
    # x depends on an A/B INTERACTION (XOR-like 2x3 mean grid): neither A alone nor B alone separates the
    # groups (each marginal averages to ~0), only the joint (A, B) key does -- a genuine composite-regime
    # win that single-column conditioning on A (or B) alone structurally cannot capture.
    """Regime conditioned fill hierarchical composite beats single column."""
    rng = np.random.default_rng(0)
    n = 6000

    a = rng.integers(0, 2, n)
    b = np.empty(n, dtype=int)
    # for A=0, B is uniform (0/1/2); for A=1, B=2 is made deliberately rare so the composite (A=1, B=2)
    # group ends up too sparse to trust directly, forcing the hierarchical fallback to the A-only median.
    for i in range(n):
        if a[i] == 0:
            b[i] = rng.choice([0, 1, 2])
        else:
            b[i] = rng.choice([0, 1, 2], p=[0.495, 0.495, 0.01])

    mean_grid = {(0, 0): -6.0, (0, 1): 0.0, (0, 2): 6.0, (1, 0): 6.0, (1, 1): 0.0, (1, 2): -6.0}
    true_x = np.array([mean_grid[(ai, bi)] for ai, bi in zip(a, b)]) + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"a": a, "b": b, "x": true_x})
    nan_mask = rng.random(n) < 0.3
    df_missing = df.copy()
    df_missing.loc[nan_mask, "x"] = np.nan

    single_filled = regime_conditioned_median_fill(df_missing, regime_col="a", feature_cols=["x"])
    composite_filled = regime_conditioned_median_fill(df_missing, regime_col="a", feature_cols=["x"], extra_regime_cols=["b"], min_group_size=25)

    single_error = float(np.mean(np.abs(single_filled.loc[nan_mask, "x"] - true_x[nan_mask])))
    composite_error = float(np.mean(np.abs(composite_filled.loc[nan_mask, "x"] - true_x[nan_mask])))

    assert composite_error < single_error * 0.6, (
        f"hierarchical composite-regime fill should recover the A/B interaction far better than A-only "
        f"conditioning: composite={composite_error:.4f} single={single_error:.4f}"
    )

    # the sparse (A=1, B=2) composite group must not be left NaN or blown up by a noisy few-sample median --
    # the hierarchical fallback to the A-only median must have kicked in for it.
    sparse_rows = nan_mask & (df["a"] == 1) & (df["b"] == 2)
    assert sparse_rows.sum() > 0
    assert not composite_filled.loc[sparse_rows, "x"].isna().any()
