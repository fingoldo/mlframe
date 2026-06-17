"""Unit + biz_value tests for test-time augmentation (`_tta.py`)."""

from __future__ import annotations

import numpy as np

from mlframe.training._tta import tta_predict, tta_predict_spread


def test_tta_noop_when_sigma_zero_or_single_sample():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))

    def f(Z):
        return Z[:, 0] * 2.0

    assert np.allclose(tta_predict(f, X, n=16, sigma_scale=0.0), f(X))
    assert np.allclose(tta_predict(f, X, n=1, sigma_scale=0.1), f(X))


def test_tta_mean_unbiased_for_linear_model():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])

    def f(Z):
        return Z @ w

    out = tta_predict(f, X, n=64, sigma_scale=0.05, seed=3)
    # Linear model: jitter averages out, TTA mean ~ clean prediction.
    assert np.corrcoef(out, f(X))[0, 1] > 0.999


def test_tta_probabilities_stay_simplex():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 3))

    def proba(Z):
        s = Z[:, 0]
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p1, p1])

    out = tta_predict(proba, X, n=32, sigma_scale=0.03, seed=5)
    assert out.shape == (500, 2)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)  # mean of simplex rows stays on the simplex


def test_tta_spread_zero_for_constant_model_positive_otherwise():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 3))

    def const(Z):
        return np.ones(Z.shape[0])

    assert np.allclose(tta_predict_spread(const, X, n=16, sigma_scale=0.1), 0.0)

    def sens(Z):
        return Z[:, 0] * 5.0

    sp = tta_predict_spread(sens, X, n=16, sigma_scale=0.1, seed=1)
    assert np.mean(sp) > 0.0


def test_biz_val_tta_improves_noisy_regression_rmse():
    """On a model whose predictions carry input-coupled noise, averaging perturbed passes reduces RMSE.

    Simulate a jittery predictor f(x)=true(x)+eps(x) where eps depends on x; TTA averaging over independent
    perturbations cancels part of eps, lowering honest RMSE vs the single clean pass. Floor the win at 2%.
    """
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.standard_normal((n, 3))
    true = X[:, 0] * 1.5 - X[:, 1]

    def jittery(Z):
        # prediction = signal + a noise term that depends on the (perturbed) input -> averaging cancels it
        return Z[:, 0] * 1.5 - Z[:, 1] + 0.8 * np.sin(13.0 * Z[:, 2])

    rmse_clean = np.sqrt(np.mean((jittery(X) - true) ** 2))
    out = tta_predict(jittery, X, n=64, sigma_scale=0.25, seed=2)
    rmse_tta = np.sqrt(np.mean((out - true) ** 2))
    assert rmse_tta <= rmse_clean * 0.98, (rmse_tta, rmse_clean)
