"""Numerical-stability regression tests for the transformer linear-algebra helpers.

Pre-fix these inverted / solved a covariance or local Hessian that, when near-singular
(collinear features) but not EXACTLY singular, slipped past the ``LinAlgError`` guard and
produced exploded / non-finite outputs. The pinv / lstsq swaps keep the output finite.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer.class_mahalanobis import _shrunk_covariance, _mahalanobis
from mlframe.feature_engineering.transformer.lda_projection import _fisher_lda
from mlframe.feature_engineering.transformer.local_classifier import _solve_weighted_linreg


def _collinear_block(n: int, rng: np.random.Generator) -> np.ndarray:
    # Two informative dims, a third = exact copy of the first -> rank-deficient cov.
    """Helper: Collinear block."""
    a = rng.standard_normal((n, 1)).astype(np.float64)
    b = rng.standard_normal((n, 1)).astype(np.float64)
    return np.hstack([a, b, a + 1e-9 * rng.standard_normal((n, 1))]).astype(np.float32)


def test_shrunk_covariance_collinear_inverse_finite():
    """Shrunk covariance collinear inverse finite."""
    rng = np.random.default_rng(0)
    X = _collinear_block(64, rng)
    mean, inv_cov = _shrunk_covariance(X)
    assert np.all(np.isfinite(inv_cov)), "inv covariance must be finite on collinear input"
    d2 = _mahalanobis(X, mean, inv_cov)
    assert np.all(np.isfinite(d2)), "Mahalanobis distances must be finite on collinear input"


def test_fisher_lda_collinear_direction_finite():
    """Fisher lda collinear direction finite."""
    rng = np.random.default_rng(1)
    X_pos = _collinear_block(48, rng) + 1.0
    X_neg = _collinear_block(48, rng)
    w, c = _fisher_lda(X_pos, X_neg)
    assert np.all(np.isfinite(w)), "LDA direction must be finite on collinear input"
    assert np.isfinite(c)


def test_local_weighted_linreg_collinear_beta_finite():
    """Local weighted linreg collinear beta finite."""
    rng = np.random.default_rng(2)
    X_local = _collinear_block(40, rng)
    y_local = rng.standard_normal(40).astype(np.float32)
    w = np.ones(40, dtype=np.float32)
    x_query = X_local[0]
    # ridge=0 forces the ill-conditioned Hessian (no Tikhonov rescue), so lstsq vs solve matters.
    pred, resid_std, coef_norm, r2 = _solve_weighted_linreg(X_local, y_local, w, x_query, ridge=0.0)
    for v in (pred, resid_std, coef_norm, r2):
        assert np.isfinite(v), f"local weighted regression produced non-finite {v}"
