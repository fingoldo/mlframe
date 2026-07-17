"""Unit + biz_value tests for test-time augmentation (`_tta.py`)."""

from __future__ import annotations

import numpy as np

from mlframe.training._tta import tta_predict, tta_predict_spread, tta_point_mean_spread


def test_tta_noop_when_sigma_zero_or_single_sample():
    """Tta noop when sigma zero or single sample."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))

    def f(Z):
        """Deterministic linear stand-in predictor: doubles the first column."""
        return Z[:, 0] * 2.0

    assert np.allclose(tta_predict(f, X, n=16, sigma_scale=0.0), f(X))
    assert np.allclose(tta_predict(f, X, n=1, sigma_scale=0.1), f(X))


def test_tta_mean_unbiased_for_linear_model():
    """Tta mean unbiased for linear model."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])

    def f(Z):
        """Linear stand-in predictor: a fixed weight vector dotted with the input."""
        return Z @ w

    out = tta_predict(f, X, n=64, sigma_scale=0.05, seed=3)
    # Linear model: jitter averages out, TTA mean ~ clean prediction.
    assert np.corrcoef(out, f(X))[0, 1] > 0.999


def test_tta_probabilities_stay_simplex():
    """Tta probabilities stay simplex."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 3))

    def proba(Z):
        """Proba."""
        s = Z[:, 0]
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p1, p1])

    out = tta_predict(proba, X, n=32, sigma_scale=0.03, seed=5)
    assert out.shape == (500, 2)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)  # mean of simplex rows stays on the simplex


def test_tta_spread_zero_for_constant_model_positive_otherwise():
    """Tta spread zero for constant model positive otherwise."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 3))

    def const(Z):
        """Const."""
        return np.ones(Z.shape[0])

    assert np.allclose(tta_predict_spread(const, X, n=16, sigma_scale=0.1), 0.0)

    def sens(Z):
        """Sens."""
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
        """Jittery."""
        return Z[:, 0] * 1.5 - Z[:, 1] + 0.8 * np.sin(13.0 * Z[:, 2])

    rmse_clean = np.sqrt(np.mean((jittery(X) - true) ** 2))
    out = tta_predict(jittery, X, n=64, sigma_scale=0.25, seed=2)
    rmse_tta = np.sqrt(np.mean((out - true) ** 2))
    assert rmse_tta <= rmse_clean * 0.98, (rmse_tta, rmse_clean)


def _identity_stack(Z):
    """Identity stack."""
    return Z.copy()


def test_tta_reproducible_under_fixed_seed():
    """A fixed seed gives byte-identical output across repeated calls (SeedSequence.spawn determinism)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    a = tta_predict(_identity_stack, X, n=8, sigma_scale=0.1, seed=123)
    b = tta_predict(_identity_stack, X, n=8, sigma_scale=0.1, seed=123)
    np.testing.assert_array_equal(a, b)


def test_tta_different_seeds_diverge():
    """Two builds with DIFFERENT seeds draw different noise -> different TTA means (independent diversity)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    a = tta_predict(_identity_stack, X, n=8, sigma_scale=0.1, seed=1)
    c = tta_predict(_identity_stack, X, n=8, sigma_scale=0.1, seed=2)
    assert not np.allclose(a, c)


def test_tta_passes_are_mutually_diverse():
    """The n-1 jittered passes must use INDEPENDENT noise streams, not a repeated draw.

    Regression for the per-member-seed fix: spawn(n-1) yields distinct child streams, so per-pass
    predictions differ from each other (a positive TTA spread on a sensitive model).
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))

    def sens(Z):
        """Sens."""
        return Z[:, 0] * 3.0 + Z[:, 1]

    sp = tta_predict_spread(sens, X, n=12, sigma_scale=0.2, seed=9)
    assert np.mean(sp) > 0.0  # passes are not identical


def test_tta_point_mean_spread_matches_standalone_helpers():
    """The fused streaming helper uses the SAME spawn scheme, so mean/spread match the standalone functions."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 3))

    def f(Z):
        """Deterministic linear stand-in predictor combining two input columns."""
        return Z[:, 0] * 2.0 - Z[:, 2]

    mean_std = tta_predict(f, X, n=10, sigma_scale=0.15, seed=7)
    spread_std = tta_predict_spread(f, X, n=10, sigma_scale=0.15, seed=7)
    _, mean_f, spread_f = tta_point_mean_spread(f, X, n=10, sigma_scale=0.15, seed=7)
    np.testing.assert_allclose(mean_f, mean_std, atol=1e-9)
    np.testing.assert_allclose(spread_f, spread_std, atol=1e-9)
