"""Unit + biz_value tests for the noise-injected ensemble (`_noise_ensemble.py`, Workstream B3)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._noise_ensemble import NoiseAugmentedEnsemble


def test_predict_shape_and_classes_passthrough():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = (X[:, 0] > 0).astype(int)
    ens = NoiseAugmentedEnsemble(LogisticRegression(max_iter=200), k=4, sigma_scale=0.05).fit(X, y)
    assert ens.predict(X).shape == (300,)
    assert ens.predict_proba(X).shape == (300, 2)
    assert list(ens.classes_) == [0, 1]


def test_picklable():
    import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 3))
    y = X @ np.array([1.0, -1.0, 0.5])
    ens = NoiseAugmentedEnsemble(Ridge(), k=3).fit(X, y)
    ens2 = pickle.loads(pickle.dumps(ens))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert np.allclose(ens.predict(X), ens2.predict(X))


def test_biz_val_noise_ensemble_helps_high_variance_1nn():
    """Noise-injection ensembling lowers honest-test RMSE on a HIGH-VARIANCE non-tree learner (1-NN).

    1-NN overfits the noisy training labels; training k copies on per-feature-jittered inputs and averaging
    smooths the prediction surface (input-noise ~ Tikhonov). Measured ratio ~0.66 (s=0.2, k=20); floor the
    win at 15%. This is the legitimate scope of B3 -- variance-prone non-tree models.
    """
    from sklearn.neighbors import KNeighborsRegressor

    rng = np.random.default_rng(3)
    n, p = 800, 6
    X = rng.standard_normal((n, p))
    Xte = rng.standard_normal((3000, p))

    def tgt(Z):
        return np.sin(Z[:, 0]) + 0.5 * Z[:, 1]

    y = tgt(X) + 0.5 * rng.standard_normal(n)
    yte = tgt(Xte)

    rmse_1nn = np.sqrt(np.mean((KNeighborsRegressor(n_neighbors=1).fit(X, y).predict(Xte) - yte) ** 2))
    ens = NoiseAugmentedEnsemble(KNeighborsRegressor(n_neighbors=1), k=20, sigma_scale=0.2, seed=1).fit(X, y)
    rmse_ens = np.sqrt(np.mean((ens.predict(Xte) - yte) ** 2))
    assert rmse_ens <= rmse_1nn * 0.85, (rmse_ens, rmse_1nn)


def test_noise_ensemble_does_not_help_low_variance_ols():
    """Documented negative result (REJECTED!=DELETED): on low-variance OLS (n>p, low label noise) input-noise
    HURTS via errors-in-variables attenuation -- so B3 is scoped to high-variance non-tree learners, not OLS."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(3)
    n, p = 220, 80
    X = rng.standard_normal((n, p))
    Xte = rng.standard_normal((2000, p))
    w = np.zeros(p)
    w[:5] = [2.0, -1.5, 1.0, -0.8, 0.5]
    y = X @ w + 0.1 * rng.standard_normal(n)
    yte = Xte @ w + 0.1 * rng.standard_normal(2000)
    rmse_ols = np.sqrt(np.mean((LinearRegression().fit(X, y).predict(Xte) - yte) ** 2))
    rmse_ens = np.sqrt(np.mean((NoiseAugmentedEnsemble(LinearRegression(), k=15, sigma_scale=0.5, seed=1).fit(X, y).predict(Xte) - yte) ** 2))
    assert rmse_ens > rmse_ols  # pins the negative result so nobody enables B3 for low-variance OLS expecting a win


def test_noise_ensemble_members_are_diverse_and_reproducible():
    """Per-member spawn: clones get INDEPENDENT noise (diverse members) and a fixed seed is reproducible.

    Regression for the per-member-seed fix. Two builds with the same seed must produce identical
    predictions; the individual members within a build must not be all-identical (independent streams).
    """
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsRegressor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = X[:, 0]

    e1 = NoiseAugmentedEnsemble(KNeighborsRegressor(n_neighbors=1), k=5, sigma_scale=0.3, seed=11).fit(X, y)
    e2 = NoiseAugmentedEnsemble(KNeighborsRegressor(n_neighbors=1), k=5, sigma_scale=0.3, seed=11).fit(X, y)
    np.testing.assert_allclose(e1.predict(X), e2.predict(X))  # reproducible under fixed seed

    # Members trained on independent noise draws are not all identical.
    member_preds = [est.predict(X) for est in e1.estimators_]
    assert not all(np.allclose(member_preds[0], mp) for mp in member_preds[1:])

    # A different seed diverges.
    e3 = NoiseAugmentedEnsemble(KNeighborsRegressor(n_neighbors=1), k=5, sigma_scale=0.3, seed=12).fit(X, y)
    assert not np.allclose(e1.predict(X), e3.predict(X))
