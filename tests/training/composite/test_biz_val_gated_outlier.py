"""biz_value test for ``training.composite.GatedOutlierEstimator``.

The win: on a zero-inflated target (a discrete point mass at 0 for "no purchase" rows, plus a continuous
spend distribution elsewhere), a single regressor fit across both regimes is pulled toward the point-mass
value on the continuous rows and away from it on the point-mass rows. Splitting into a classifier gate ("is
this the point mass?") plus a regressor fit only on the continuous rows, blended by the classifier's
predicted probability, should recover a materially lower test MSE than the single-regressor baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.training.composite import GatedOutlierEstimator


def _make_zero_inflated_dataset(n: int, seed: int):
    """Make zero inflated dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    is_purchase = rng.random(n) < (0.1 + 0.8 * (X[:, 0] > 0))
    y = np.zeros(n)
    y[is_purchase] = np.clip(100 + 20 * X[is_purchase, 1] + rng.normal(0, 5, is_purchase.sum()), 1, None)
    return X, y


def test_biz_val_gated_outlier_beats_single_regressor_mse_on_zero_inflated_target():
    """Biz val gated outlier beats single regressor mse on zero inflated target."""
    X, y = _make_zero_inflated_dataset(n=5000, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_mse = mean_squared_error(y_test, baseline.predict(X_test))

    gated = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0)
    gated.fit(X_train, y_train)
    gated_mse = mean_squared_error(y_test, gated.predict(X_test))

    improvement = 1.0 - gated_mse / baseline_mse
    assert (
        improvement > 0.15
    ), f"expected >15% MSE reduction over a single regressor, got {improvement:.4f} (baseline={baseline_mse:.2f}, gated={gated_mse:.2f})"


def test_gated_outlier_point_mass_rate_matches_training_data():
    """Gated outlier point mass rate matches training data."""
    X, y = _make_zero_inflated_dataset(n=2000, seed=1)
    gated = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0)
    gated.fit(X, y)
    assert abs(gated.point_mass_rate_ - (y == 0).mean()) < 1e-9


def test_gated_outlier_all_point_mass_predicts_constant():
    """Gated outlier all point mass predicts constant."""
    X = np.random.default_rng(2).normal(size=(50, 3))
    y = np.zeros(50)
    gated = GatedOutlierEstimator(regressor=LinearRegression(), point_mass_value=0.0)
    gated.fit(X, y)
    pred = gated.predict(X)
    assert np.allclose(pred, 0.0)


def _make_smooth_boundary_dataset(n: int, seed: int):
    """Zero-inflated target with a smooth logistic point-mass boundary (vs. the hard-cutoff dataset above).

    A shallow ``RandomForestClassifier`` gate on this boundary is systematically miscalibrated (tree-ensemble
    probabilities are biased toward the extremes even though the ranking/AUC is fine) -- exactly the case
    ``calibrate_classifier`` targets, since the blend's error is directly proportional to the gate probability's
    calibration error, not just its ranking.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    is_purchase = rng.random(n) < p
    y = np.zeros(n)
    y[is_purchase] = np.clip(100 + 20 * X[is_purchase, 2] + rng.normal(0, 5, is_purchase.sum()), 1, None)
    return X, y


def test_biz_val_gated_outlier_calibrate_classifier_beats_uncalibrated_gate_mse():
    """Biz val gated outlier calibrate classifier beats uncalibrated gate mse."""
    X, y = _make_smooth_boundary_dataset(n=6000, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    def _fit_and_mse(calibrate: bool) -> float:
        """Fit and mse."""
        gated = GatedOutlierEstimator(
            regressor=LinearRegression(),
            classifier=RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0),
            point_mass_value=0.0,
            calibrate_classifier=calibrate,
            calibration_cv=3,
        )
        gated.fit(X_train, y_train)
        return float(mean_squared_error(y_test, gated.predict(X_test)))

    uncalibrated_mse = _fit_and_mse(calibrate=False)
    calibrated_mse = _fit_and_mse(calibrate=True)

    improvement = 1.0 - calibrated_mse / uncalibrated_mse
    assert improvement > 0.02, (
        f"expected >2% MSE reduction from calibrating a miscalibrated tree-ensemble gate, got {improvement:.4f} "
        f"(uncalibrated={uncalibrated_mse:.2f}, calibrated={calibrated_mse:.2f})"
    )


def test_gated_outlier_calibrate_classifier_default_off_is_bit_identical_to_prior_behavior():
    """Gated outlier calibrate classifier default off is bit identical to prior behavior."""
    X, y = _make_zero_inflated_dataset(n=1500, seed=4)
    baseline = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0)
    baseline.fit(X, y)
    default_explicit_off = GatedOutlierEstimator(
        regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0, calibrate_classifier=False
    )
    default_explicit_off.fit(X, y)

    assert type(baseline.classifier_).__name__ == "LogisticRegression"
    assert type(default_explicit_off.classifier_).__name__ == "LogisticRegression"
    np.testing.assert_array_equal(baseline.predict(X), default_explicit_off.predict(X))


def test_gated_outlier_custom_blend_value():
    """Gated outlier custom blend value."""
    X, y = _make_zero_inflated_dataset(n=1000, seed=3)
    gated_zero = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0)
    gated_zero.fit(X, y)
    gated_neg = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000), point_mass_value=0.0, blend_value=-100.0)
    gated_neg.fit(X, y)
    # Swapping blend_value from the point_mass_value default (0) to a large negative constant should pull
    # predictions down proportionally to each row's point-mass probability, without touching the regressor.
    p = gated_zero.predict_proba_point_mass(X)
    confident_point_mass = p > 0.9
    assert confident_point_mass.any()
    assert (gated_neg.predict(X)[confident_point_mass] < gated_zero.predict(X)[confident_point_mass]).all()
