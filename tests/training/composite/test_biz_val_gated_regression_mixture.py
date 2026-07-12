"""biz_value test for ``training.composite.GatedRegressionMixture``.

The win: a rare outlier subpopulation has an extra target offset driven by a hidden "severity" that also
correlates with the gate classifier's own predicted probability (higher confidence of being an outlier ~
higher severity). A single global regressor can't represent the subpopulation-specific offset at all. A
regressor routed to the high-probability branch WITHOUT the gate's probability as an input feature can't
recover the severity-dependent offset either (severity isn't directly observable). Stacking the gate's OOF
probability in as an extra feature for the routed regressor lets it partially recover that hidden signal --
mirroring the Elo Merchant 5th place's routing + stacked-gate-feature + per-branch-weighting technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

from mlframe.training.composite import GatedRegressionMixture


def _make_outlier_severity_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    is_outlier = rng.random(n) < 0.15
    x = rng.normal(size=n)
    severity = np.where(is_outlier, rng.uniform(0, 1, n), 0.0)
    evidence = severity + rng.normal(scale=0.3, size=n)  # observable proxy correlated with severity
    y = x * 1.0 + np.where(is_outlier, 5.0 * severity, 0.0) + rng.normal(scale=0.3, size=n)
    X = pd.DataFrame({"x": x, "evidence": evidence})
    return X, y, is_outlier


def test_biz_val_gated_regression_mixture_beats_single_global_regressor_mse():
    X, y, is_outlier = _make_outlier_severity_dataset(3000, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:2000], perm[2000:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    baseline = LinearRegression().fit(X_train, y_train)
    mse_baseline = mean_squared_error(y_test, baseline.predict(X_test))

    mixture = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=True, n_splits=5, random_state=0,
    )
    mixture.fit(X_train, y_train, is_outlier[train_idx])
    mse_mixture = mean_squared_error(y_test, mixture.predict(X_test))

    improvement = 1.0 - mse_mixture / mse_baseline
    assert improvement > 0.15, f"expected >15% MSE reduction vs. a single global regressor, got {improvement:.4f} (baseline={mse_baseline:.4f}, mixture={mse_mixture:.4f})"


def test_gated_regression_mixture_gate_feature_ablation_improves_high_branch():
    """The stacked gate-probability feature should give a real additional lift over routing alone (hard
    threshold routing without exposing the gate probability to the regressor)."""
    X, y, is_outlier = _make_outlier_severity_dataset(3000, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:2000], perm[2000:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    mixture_with_feature = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=True, n_splits=5, random_state=0,
    )
    mixture_with_feature.fit(X_train, y_train, is_outlier[train_idx])
    mse_with = mean_squared_error(y_test, mixture_with_feature.predict(X_test))

    mixture_without_feature = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=False, n_splits=5, random_state=0,
    )
    mixture_without_feature.fit(X_train, y_train, is_outlier[train_idx])
    mse_without = mean_squared_error(y_test, mixture_without_feature.predict(X_test))

    assert mse_with < mse_without, f"expected the stacked gate feature to improve MSE, got with={mse_with:.4f} without={mse_without:.4f}"


def _make_boundary_transition_dataset(n: int, seed: int):
    """A single SMOOTH ground-truth relation (no true discontinuity) over a score ``s`` used both as the
    gate-classifier's driving feature (label ``s >= 0.5``) and as the regression feature. Piecewise-linear
    branch regressors fit independently on the low (``s<0.5``) and high (``s>=0.5``) truncated ranges of a
    non-linear (``sin``) relation diverge from each other right at the boundary -- a real prediction
    discontinuity introduced purely by hard threshold routing, not present in the underlying truth."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, n)
    is_high = s >= 0.5
    y = 3.0 * s + 2.0 * np.sin(5.0 * s) + rng.normal(scale=0.05, size=n)
    X = pd.DataFrame({"s": s})
    return X, y, is_high


def test_biz_val_gated_regression_mixture_soft_routing_reduces_boundary_error():
    """Soft routing (probability-weighted blend near the gate threshold) must reduce prediction error for
    rows in the transition zone versus hard 0/1 routing, without changing default (``soft_routing=False``)
    behavior at all -- checked separately below for bit-identity."""
    X, y, is_high = _make_boundary_transition_dataset(4000, seed=3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:3000], perm[3000:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    mixture = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=False, n_splits=5, random_state=0,
        soft_routing=False, soft_routing_bandwidth=0.15,
    )
    mixture.fit(X_train, y_train, is_high[train_idx])

    gate_proba = mixture._predict_proba_1(mixture.gate_model_, X_test)
    band = (gate_proba >= 0.5 - 0.15) & (gate_proba <= 0.5 + 0.15)
    assert band.sum() >= 30, f"expected a meaningfully-sized boundary band, got {band.sum()} rows"

    pred_hard = mixture.predict(X_test)
    mse_hard_band = mean_squared_error(y_test[band], pred_hard[band])

    mixture.soft_routing = True
    pred_soft = mixture.predict(X_test)
    mse_soft_band = mean_squared_error(y_test[band], pred_soft[band])

    improvement = 1.0 - mse_soft_band / mse_hard_band
    assert improvement > 0.05, (
        f"expected >5% boundary-band MSE reduction from soft routing, got {improvement:.4f} "
        f"(hard={mse_hard_band:.4f}, soft={mse_soft_band:.4f})"
    )

    # Rows OUTSIDE the band must be untouched by soft routing (single-branch hard route either way).
    assert np.allclose(pred_hard[~band], pred_soft[~band])


def test_gated_regression_mixture_soft_routing_default_off_is_bit_identical():
    """``soft_routing`` defaults to False -- predict() output must be EXACTLY unchanged from before this
    feature existed (same code path: single-branch hard route for every row, empty blend band)."""
    X, y, is_high = _make_boundary_transition_dataset(500, seed=5)

    m_explicit_off = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=False, n_splits=3, random_state=0, soft_routing=False,
    )
    m_explicit_off.fit(X, y, is_high)
    pred_explicit_off = m_explicit_off.predict(X)

    m_default = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=False, n_splits=3, random_state=0,
    )
    m_default.fit(X, y, is_high)
    pred_default = m_default.predict(X)

    assert np.array_equal(pred_explicit_off, pred_default)


def test_gated_regression_mixture_branch_sample_weight_changes_fit():
    """A UNIFORM per-branch sample_weight multiplier is mathematically inert for plain OLS (scaling every
    row's weight by the same constant doesn't change the normal equations) -- use Ridge, where the weight
    scale interacts with the regularization strength, to give the downweighting knob observable teeth."""
    from sklearn.linear_model import Ridge

    X, y, is_outlier = _make_outlier_severity_dataset(500, seed=2)
    m1 = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=Ridge(alpha=5.0), high_regressor=Ridge(alpha=5.0),
        branch_sample_weight={"low": 1.0, "high": 1.0}, n_splits=3, random_state=0,
    )
    m1.fit(X, y, is_outlier)
    m2 = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=Ridge(alpha=5.0), high_regressor=Ridge(alpha=5.0),
        branch_sample_weight={"low": 0.1, "high": 1.0}, n_splits=3, random_state=0,
    )
    m2.fit(X, y, is_outlier)
    if "low" in m1.branch_models_ and "low" in m2.branch_models_:
        assert not np.allclose(m1.branch_models_["low"].coef_, m2.branch_models_["low"].coef_)
