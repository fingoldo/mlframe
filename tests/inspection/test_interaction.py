"""Unit + biz_value tests for Friedman & Popescu H-statistic (PZAD interpretability)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.inspection import friedman_h_statistic, pairwise_interaction_strength


class _AdditiveModel:
    """f(x) = 3*x0 + 2*x1 (no interaction) -> H(0,1) must be ~0."""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return 3.0 * X[:, 0] + 2.0 * X[:, 1]


class _ProductModel:
    """f(x) = x0 * x1 (pure interaction, no main effect at mean 0) -> H(0,1) must be high."""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, 0] * X[:, 1]


def _data(seed=0, n=1500, d=4):
    return np.random.default_rng(seed).normal(size=(n, d))


# ---------------------------------------------------------------- unit
def test_additive_model_has_near_zero_interaction():
    X = _data()
    h = friedman_h_statistic(_AdditiveModel(), X, (0, 1))
    assert h < 0.1, f"purely additive model should have H~0, got {h:.3f}"


def test_product_model_has_strong_interaction():
    X = _data()
    h = friedman_h_statistic(_ProductModel(), X, (0, 1))
    assert h > 0.5, f"pure product interaction should have high H, got {h:.3f}"


def test_irrelevant_feature_pair_low():
    # x2 does not enter the product model -> H(0,2) low even though x0 interacts with x1
    X = _data()
    h02 = friedman_h_statistic(_ProductModel(), X, (0, 2))
    assert h02 < 0.15, f"a feature the model ignores should have low interaction, got {h02:.3f}"


def test_symmetry():
    X = _data()
    h01 = friedman_h_statistic(_ProductModel(), X, (0, 1))
    h10 = friedman_h_statistic(_ProductModel(), X, (1, 0))
    assert abs(h01 - h10) < 1e-9


def test_same_feature_raises():
    with pytest.raises(ValueError):
        friedman_h_statistic(_AdditiveModel(), _data(), (1, 1))


def test_constant_feature_zero():
    X = _data()
    X[:, 2] = 5.0  # constant column
    h = friedman_h_statistic(_ProductModel(), X, (0, 2))
    assert h == 0.0


def test_matrix_shape_and_symmetry():
    X = _data()
    M = pairwise_interaction_strength(_ProductModel(), X, [0, 1, 2])
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    assert M[0, 1] > M[0, 2]  # x0-x1 interact; x0-x2 do not


# ---------------------------------------------------------------- biz_value
def test_biz_val_h_statistic_ranks_true_interaction_above_additive_on_fitted_model():
    """On a REAL fitted model: target has a genuine x0*x1 interaction plus an additive x2 main effect. The H-statistic
    must rank the interacting pair (0,1) far above the additive-only pair (0,2) and (1,2), recovering which pair a
    feature-engineer should build an explicit interaction term for. Uses a gradient-boosted model, not the analytic
    predictor -- so the test guards the whole PDP->H pipeline against regressions."""
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(1)
    n = 3000
    X = rng.uniform(-2, 2, size=(n, 3))
    y = X[:, 0] * X[:, 1] + 1.5 * X[:, 2] + 0.05 * rng.normal(size=n)  # interaction(0,1) + additive main(2)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(X, y)

    M = pairwise_interaction_strength(model, X, [0, 1, 2], sample=1500)
    h01, h02, h12 = M[0, 1], M[0, 2], M[1, 2]

    assert h01 >= 0.30, f"the true x0*x1 interaction should register H>=0.30, got {h01:.3f}"
    assert h01 >= h02 + 0.15 and h01 >= h12 + 0.15, f"interacting pair {h01:.3f} must top additive pairs ({h02:.3f}, {h12:.3f})"
