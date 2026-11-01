"""The GIL-free Newton kernel of the accuracy-gate probe reaches the optimum of its NumPy twin."""

import numpy as np

from mlframe.feature_selection.filters._fe_accuracy_gate_newton import fit_logistic_ridge_newton, fit_logistic_ridge_newton_numpy


def test_kernel_matches_numpy_twin():
    """Weights and intercept agree on a noisy, an imbalanced and a nearly separable problem."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    cases = [
        (X, (X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n) > 0).astype(float)),
        (X, (X[:, 2] + rng.normal(size=n) > 2.0).astype(float)),
        (X, (X[:, 0] > 0).astype(float)),
    ]
    for Xc, yc in cases:
        got, want = fit_logistic_ridge_newton(Xc, yc), fit_logistic_ridge_newton_numpy(Xc, yc)
        assert (got is None) == (want is None)
        if got is not None:
            np.testing.assert_allclose(got[0], want[0], rtol=1e-6, atol=1e-8)
            assert abs(got[1] - want[1]) < 1e-6


def test_constant_labels_defer_to_the_caller():
    """A single-class fit has no finite optimum with a free intercept: both forms return None or agree."""
    X = np.random.default_rng(1).normal(size=(500, 2))
    y = np.ones(500)
    assert (fit_logistic_ridge_newton(X, y) is None) == (fit_logistic_ridge_newton_numpy(X, y) is None)
