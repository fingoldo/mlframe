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
    """Helper: Annuity constraint fn."""
    if rate <= 0:
        return np.full(len(df), np.inf)
    implied_annuity = df["amount"].to_numpy() * rate / (1 - (1 + rate) ** (-df["n"].to_numpy()))
    return np.asarray(implied_annuity - df["annuity"].to_numpy())


def _make_loan_data(n: int, seed: int):
    """Helper: Make loan data."""
    rng = np.random.default_rng(seed)
    true_rate = rng.uniform(0.005, 0.03, n)
    duration = rng.choice([12, 24, 36, 48, 60], n).astype(float)
    amount = rng.uniform(50000, 300000, n)
    annuity = amount * true_rate / (1 - (1 + true_rate) ** (-duration)) + rng.normal(scale=5, size=n)
    df = pd.DataFrame({"amount": amount, "n": duration, "annuity": annuity})
    return df, true_rate


def test_biz_val_recovery_features_beat_raw_column_baseline():
    """Biz val recovery features beat raw column baseline."""
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

    assert rmse_recovery < rmse_raw * 0.5, (
        f"expected candidate-summary features to beat the raw-column linear baseline by >=50% RMSE, got recovery={rmse_recovery:.5f} raw={rmse_raw:.5f}"
    )


def test_latent_parameter_recovery_features_hand_computed():
    """Latent parameter recovery features hand computed."""
    df = pd.DataFrame({"x": [10.0]})

    def constraint_fn(df: pd.DataFrame, candidate: float) -> np.ndarray:
        """Constraint fn."""
        return df["x"].to_numpy() - candidate

    feats = latent_parameter_recovery_features(df, candidate_grid=[8.0, 10.0, 12.0], constraint_fn=constraint_fn, tolerance=0.5)
    assert feats.loc[0, "latent_param_n_candidates"] == 1
    assert feats.loc[0, "latent_param_mean"] == 10.0


def test_latent_parameter_recovery_features_no_consistent_candidate_is_nan():
    """Latent parameter recovery features no consistent candidate is nan."""
    df = pd.DataFrame({"x": [100.0]})

    def constraint_fn(df: pd.DataFrame, candidate: float) -> np.ndarray:
        """Constraint fn."""
        return df["x"].to_numpy() - candidate

    feats = latent_parameter_recovery_features(df, candidate_grid=[1.0, 2.0, 3.0], constraint_fn=constraint_fn, tolerance=0.1)
    assert feats.loc[0, "latent_param_n_candidates"] == 0
    assert np.isnan(feats.loc[0, "latent_param_mean"])


def test_latent_parameter_recovery_features_weight_fn_omitted_is_bit_identical():
    # weight_fn is strictly opt-in: omitting it must reproduce the pre-extension uniform-candidate output.
    """Latent parameter recovery features weight fn omitted is bit identical."""
    df, _ = _make_loan_data(n=300, seed=3)
    grid = np.arange(0.002, 0.05, 0.0005)
    baseline = latent_parameter_recovery_features(df, grid, _annuity_constraint_fn, tolerance=150.0)
    extended = latent_parameter_recovery_features(df, grid, _annuity_constraint_fn, tolerance=150.0, weight_fn=None)
    pd.testing.assert_frame_equal(baseline, extended)


def test_biz_val_recovery_features_weight_fn_beats_uniform_under_nonuniform_prior():
    # The true rate is drawn from a strongly non-uniform prior (concentrated near 0.008), but the observed
    # annuity/amount/duration relation alone leaves several grid candidates consistent per row within
    # `tolerance`. A caller who KNOWS the prior (e.g. from a portfolio-level rate distribution) should recover
    # a summary statistic closer to the true rate by down-weighting implausible candidates, versus treating
    # every constraint-satisfying candidate as equally likely.
    """Biz val recovery features weight fn beats uniform under nonuniform prior."""
    rng = np.random.default_rng(7)
    n = 600
    true_rate = rng.choice([0.008, 0.02, 0.04], size=n, p=[0.7, 0.2, 0.1]) + rng.normal(scale=0.0005, size=n)
    duration = rng.choice([12, 24, 36, 48, 60], n).astype(float)
    amount = rng.uniform(50000, 300000, n)
    annuity = amount * true_rate / (1 - (1 + true_rate) ** (-duration)) + rng.normal(scale=200, size=n)
    df = pd.DataFrame({"amount": amount, "n": duration, "annuity": annuity})

    # coarse grid + loose tolerance -> ~20-40 grid points survive the constraint per row (verified via
    # `latent_param_n_candidates`), so the prior actually has candidates to discriminate between.
    grid = np.arange(0.002, 0.05, 0.001)

    def _rate_prior_weight(df: pd.DataFrame, candidate: float) -> np.ndarray:
        # a Gaussian prior over the rate, centered where the true generative process concentrates mass.
        """Helper: Rate prior weight."""
        density = np.exp(-0.5 * ((candidate - 0.0085) / 0.01) ** 2)
        return np.full(len(df), density)

    uniform_feats = latent_parameter_recovery_features(df, grid, _annuity_constraint_fn, tolerance=2000.0)
    weighted_feats = latent_parameter_recovery_features(df, grid, _annuity_constraint_fn, tolerance=2000.0, weight_fn=_rate_prior_weight)

    assert (weighted_feats["latent_param_n_candidates"] > 0).all()

    mae_uniform = float(np.mean(np.abs(uniform_feats["latent_param_mean"].to_numpy() - true_rate)))
    mae_weighted = float(np.mean(np.abs(weighted_feats["latent_param_mean"].to_numpy() - true_rate)))

    assert mae_weighted < mae_uniform * 0.90, (
        f"expected prior-weighted mean to beat uniform-candidate mean by >=15% MAE, got weighted={mae_weighted:.6f} uniform={mae_uniform:.6f}"
    )
