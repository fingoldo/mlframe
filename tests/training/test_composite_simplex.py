"""Unit + biz_value tests for CompositeSimplexEstimator (compositional targets)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from mlframe.training.composite.simplex import (
    CompositeSimplexEstimator,
    aitchison_distance,
    alr_forward,
    alr_inverse,
    ilr_forward,
    ilr_inverse,
    multiplicative_zero_replacement,
    _ilr_basis,
)


def _make_dirichlet_data(n=600, k=4, n_feat=5, seed=0):
    """Synthetic: composition driven by a softmax of linear functions of X."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    W = rng.normal(size=(n_feat, k))
    logits = X @ W
    base = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = base / base.sum(axis=1, keepdims=True)
    # Dirichlet sampling around the structured probs => noisy but valid compositions.
    y = np.array([rng.dirichlet(p * 40 + 0.5) for p in probs])
    return X, y


# ---- forward/inverse math --------------------------------------------------


def test_ilr_basis_orthonormal():
    for k in (2, 3, 5, 8):
        v = _ilr_basis(k)
        assert v.shape == (k, k - 1)
        np.testing.assert_allclose(v.T @ v, np.eye(k - 1), atol=1e-12)
        # columns sum to zero (live in the clr hyperplane)
        np.testing.assert_allclose(v.sum(axis=0), 0.0, atol=1e-12)


def test_alr_round_trip():
    rng = np.random.default_rng(1)
    y = rng.dirichlet(np.ones(5), size=50)
    z = alr_forward(y, ref=2)
    assert z.shape == (50, 4)
    y_back = alr_inverse(z, ref=2, k=5)
    np.testing.assert_allclose(y_back, y, atol=1e-10)


def test_ilr_round_trip():
    rng = np.random.default_rng(2)
    y = rng.dirichlet(np.ones(6), size=40)
    basis = _ilr_basis(6)
    z = ilr_forward(y, basis)
    assert z.shape == (40, 5)
    y_back = ilr_inverse(z, basis)
    np.testing.assert_allclose(y_back, y, atol=1e-10)


def test_zero_replacement_preserves_closure_and_positivity():
    y = np.array([[0.0, 0.5, 0.5], [0.2, 0.0, 0.8], [0.3, 0.3, 0.4]])
    out = multiplicative_zero_replacement(y, delta=1e-4)
    assert (out > 0).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # untouched row stays identical
    np.testing.assert_allclose(out[2], y[2], atol=1e-12)


# ---- estimator contract ----------------------------------------------------


@pytest.mark.parametrize("transform", ["ilr", "alr"])
def test_predict_valid_composition(transform):
    X, y = _make_dirichlet_data(k=4)
    est = CompositeSimplexEstimator(Ridge(), transform=transform).fit(X, y)
    p = est.predict(X)
    assert p.shape == (X.shape[0], 4)
    assert (p >= 0).all(), "parts must be non-negative"
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_predict_coordinates_shape():
    X, y = _make_dirichlet_data(k=5)
    est = CompositeSimplexEstimator(Ridge(), transform="ilr").fit(X, y)
    assert est.predict_coordinates(X).shape == (X.shape[0], 4)


def test_k2_reduces_to_logistic():
    # With K=2 alr coordinate is log(y1/y0); inverse is the logistic of the
    # predicted log-odds. Verify the inverse matches sigmoid by construction.
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20, 1))
    p = alr_inverse(z, ref=0, k=2)  # ref=0 -> coordinate is log(y1/y0)
    # part 1 share should equal sigmoid(z) (logistic)
    sig = 1.0 / (1.0 + np.exp(-z[:, 0]))
    np.testing.assert_allclose(p[:, 1], sig, atol=1e-10)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_fit_rejects_non_composition():
    X, y = _make_dirichlet_data(k=4)
    with pytest.raises(ValueError):
        CompositeSimplexEstimator(Ridge()).fit(X, y[:, 0])  # 1-D


def test_handles_zeros_in_training_target():
    X, y = _make_dirichlet_data(k=4, seed=7)
    y[:10, 0] = 0.0
    y[:10] = y[:10] / y[:10].sum(axis=1, keepdims=True)
    est = CompositeSimplexEstimator(Ridge(), transform="ilr").fit(X, y)
    p = est.predict(X)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


# ---- biz_value -------------------------------------------------------------


def test_biz_val_simplex_beats_naive_independent_regressor():
    """Simplex composite ALWAYS yields valid compositions AND lower Aitchison
    error than a naive per-part regressor that can predict invalid compositions.

    Measured (Ridge, n=600, K=4, seed=0): simplex mean Aitchison ~ 0.83 vs naive
    ~ 1.15; naive also emits negative parts. Floor set well below the measured gap.
    """
    Xtr, ytr = _make_dirichlet_data(n=600, k=4, seed=0)
    Xte, yte = _make_dirichlet_data(n=400, k=4, seed=99)

    simplex = CompositeSimplexEstimator(Ridge(), transform="ilr").fit(Xtr, ytr)
    ps = simplex.predict(Xte)

    naive = MultiOutputRegressor(Ridge()).fit(Xtr, ytr)
    pn = np.asarray(naive.predict(Xte))

    # Validity: simplex always valid; naive demonstrably violates the simplex.
    assert (ps >= 0).all()
    np.testing.assert_allclose(ps.sum(axis=1), 1.0, atol=1e-9)
    naive_invalid = (pn < 0).any() or not np.allclose(pn.sum(axis=1), 1.0, atol=1e-6)
    assert naive_invalid, "naive independent regressor should violate the simplex constraints"

    err_simplex = aitchison_distance(ps, yte).mean()
    err_naive = aitchison_distance(pn, yte).mean()
    assert err_simplex < err_naive * 0.95, (
        f"simplex Aitchison {err_simplex:.4f} should beat naive {err_naive:.4f}"
    )
