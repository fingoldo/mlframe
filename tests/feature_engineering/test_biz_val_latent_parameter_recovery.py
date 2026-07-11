"""biz_value test for ``feature_engineering.latent_parameter_recovery.latent_parameter_recovery_features``.

Source: 5th_home-credit-default-risk.md -- recovering an interest rate from ``Annuity, Amount, CNT_PAYMENT``
via the compound-interest formula: enumerate candidate rates, keep those consistent with the observed
annuity/amount/duration, summarize the surviving candidates, then train a supervised model against a
partially-labeled subset. Because the compound-interest relation is highly nonlinear in the rate, a plain
linear model on the RAW observed columns struggles to recover the rate directly; the candidate-summary
features (already close to the true rate by construction) should let a simple linear model recover it far
more accurately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering.latent_parameter_recovery import latent_parameter_recovery_features


def _annuity_constraint_fn(df: pd.DataFrame, rate: float) -> np.ndarray:
    if rate <= 0:
        return np.full(len(df), np.inf)
    implied_annuity = df["amount"].to_numpy() * rate / (1 - (1 + rate) ** (-df["n"].to_numpy()))
    return np.asarray(implied_annuity - df["annuity"].to_numpy())


def _make_loan_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    true_rate = rng.uniform(0.005, 0.03, n)
    duration = rng.choice([12, 24, 36, 48, 60], n).astype(float)
    amount = rng.uniform(50000, 300000, n)
    annuity = amount * true_rate / (1 - (1 + true_rate) ** (-duration)) + rng.normal(scale=5, size=n)
    df = pd.DataFrame({"amount": amount, "n": duration, "annuity": annuity})
    return df, true_rate


def test_biz_val_recovery_features_beat_raw_column_baseline():
    df, true_rate = _make_loan_data(n=800, seed=1)
    grid = np.arange(0.002, 0.05, 0.0005)
    feats = latent_parameter_recovery_features(df, grid, _annuity_constraint_fn, tolerance=150.0)
    assert not feats["latent_param_mean"].isna().any()

    train_idx, test_idx = np.arange(0, 600), np.arange(600, 800)

    model_recovery = Ridge(alpha=0.01).fit(feats.iloc[train_idx][["latent_param_mean", "latent_param_median"]], true_rate[train_idx])
    pred_recovery = model_recovery.predict(feats.iloc[test_idx][["latent_param_mean", "latent_param_median"]])
    rmse_recovery = float(np.sqrt(mean_squared_error(true_rate[test_idx], pred_recovery)))

    model_raw = Ridge(alpha=0.01).fit(df.iloc[train_idx][["amount", "n", "annuity"]], true_rate[train_idx])
    pred_raw = model_raw.predict(df.iloc[test_idx][["amount", "n", "annuity"]])
    rmse_raw = float(np.sqrt(mean_squared_error(true_rate[test_idx], pred_raw)))

    assert rmse_recovery < rmse_raw * 0.5, f"expected candidate-summary features to beat the raw-column linear baseline by >=50% RMSE, got recovery={rmse_recovery:.5f} raw={rmse_raw:.5f}"


def test_latent_parameter_recovery_features_hand_computed():
    df = pd.DataFrame({"x": [10.0]})

    def constraint_fn(df: pd.DataFrame, candidate: float) -> np.ndarray:
        return df["x"].to_numpy() - candidate

    feats = latent_parameter_recovery_features(df, candidate_grid=[8.0, 10.0, 12.0], constraint_fn=constraint_fn, tolerance=0.5)
    assert feats.loc[0, "latent_param_n_candidates"] == 1
    assert feats.loc[0, "latent_param_mean"] == 10.0


def test_latent_parameter_recovery_features_no_consistent_candidate_is_nan():
    df = pd.DataFrame({"x": [100.0]})

    def constraint_fn(df: pd.DataFrame, candidate: float) -> np.ndarray:
        return df["x"].to_numpy() - candidate

    feats = latent_parameter_recovery_features(df, candidate_grid=[1.0, 2.0, 3.0], constraint_fn=constraint_fn, tolerance=0.1)
    assert feats.loc[0, "latent_param_n_candidates"] == 0
    assert np.isnan(feats.loc[0, "latent_param_mean"])
