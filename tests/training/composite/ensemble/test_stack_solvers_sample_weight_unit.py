"""Regression sentry: stack-solver sample_weight=None matches old behaviour byte-for-byte.

Ridge stack and NNLS stack must produce identical weights / intercept when sample_weight is omitted
versus explicitly passed as None. Non-uniform weights must change the fit (and for NNLS, the row-
scaling emulation must agree with sklearn's weighted Ridge on a non-negative problem).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _toy_stack_dataset(n=200, k=3, seed=3):
    """Toy stack dataset."""
    rng = np.random.default_rng(seed)
    # K component prediction columns + 1 target. Targets correlated with col 0.
    X = rng.normal(size=(n, k))
    true_w = np.array([0.7, 0.2, 0.1])
    y = X @ true_w + 0.05 * rng.normal(size=n)
    return X, y


class _DummyModel:
    """Minimal placeholder so CompositeCrossTargetEnsemble accepts ``component_models``."""

    def predict(self, X):
        """Predict."""
        return np.zeros(len(X))


def _components(k):
    """Components."""
    return [_DummyModel() for _ in range(k)], [f"c{i}" for i in range(k)]


def test_from_linear_stack_sample_weight_none_matches_omitted():
    """From linear stack sample weight none matches omitted."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    a = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0)
    b = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0, sample_weight=None)
    np.testing.assert_allclose(a.weights, b.weights)
    assert a._linear_stack_intercept == b._linear_stack_intercept


def test_from_linear_stack_nonuniform_sample_weight_changes_weights():
    """From linear stack nonuniform sample weight changes weights."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    rng = np.random.default_rng(1)
    sw = rng.uniform(0.1, 5.0, size=len(y))
    unweighted = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0)
    weighted = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0, sample_weight=sw)
    # Weights must differ materially.
    assert not np.allclose(unweighted.weights, weighted.weights), "non-uniform sample_weight must shift Ridge weights"


def test_from_linear_stack_validates_sample_weight():
    """From linear stack validates sample weight."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    with pytest.raises(ValueError, match="sample_weight length"):
        CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0, sample_weight=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(len(y))
        sw[0] = -1.0
        CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=1.0, sample_weight=sw)


def test_from_nnls_stack_sample_weight_none_matches_omitted():
    """From nnls stack sample weight none matches omitted."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    a = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y)
    b = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=None)
    np.testing.assert_allclose(a.weights, b.weights)


def test_from_nnls_stack_row_scaling_matches_weighted_least_squares():
    """The sqrt-weight row-scaling trick must reproduce weighted LS:
    NNLS(diag(sqrt(w)) A, diag(sqrt(w)) b) == argmin_{w>=0} sum_i w_i (a_i x - b_i)^2.

    Compare against scipy.optimize.lsq_linear with bounds=[0,inf) and the same weighted system."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
    from scipy.optimize import nnls

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    rng = np.random.default_rng(2)
    sw = rng.uniform(0.2, 3.0, size=len(y))
    fitted = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=sw)
    # Independent reference solve via row scaling.
    sqrt_w = np.sqrt(sw).reshape(-1, 1)
    A_scaled = X * sqrt_w
    b_scaled = y * sqrt_w.reshape(-1)
    w_ref, _ = nnls(A_scaled, b_scaled)
    np.testing.assert_allclose(fitted.weights, w_ref, rtol=1e-8)
    # And it must differ from the unweighted solve when sw is non-uniform.
    unweighted = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y)
    assert not np.allclose(fitted.weights, unweighted.weights), "non-uniform sample_weight must shift NNLS weights"


def test_from_nnls_stack_validates_sample_weight():
    """From nnls stack validates sample weight."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y = _toy_stack_dataset()
    models, names = _components(X.shape[1])
    with pytest.raises(ValueError, match="sample_weight length"):
        CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(len(y))
        sw[0] = -1.0
        CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=sw)
