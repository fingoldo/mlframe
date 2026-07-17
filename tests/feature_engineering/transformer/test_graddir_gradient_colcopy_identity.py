"""Regression: gradient_direction_agreement._gradient column-copy optimization.

The optimized ``_gradient`` perturbs one column in place and restores it from a
saved copy, instead of allocating a full (n,d) ``X.copy()`` per column. This test
pins:

1. bit-identity vs the reference per-column-copy implementation, and
2. that the input matrix is left unmutated after the call (restore is exact).

Fails on pre-fix code only for (1) if the optimization had changed numerics; the
real sensor is (2) -- pre-fix code did ``X.copy()`` so X was never touched, while
a broken restore in the optimized path would leave X mutated. Both are pinned so a
future "just mutate without restore" or a numerics-changing rewrite is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.gradient_direction_agreement import _gradient, _predict


class _LinModel:
    """Groups tests for: LinModel."""
    def __init__(self, d, seed):
        """Helper: Init  ."""
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(d).astype(np.float32)
        self.b = np.float32(rng.standard_normal())

    def predict(self, X):
        """Predict."""
        return (X @ self.w + self.b).astype(np.float32)


def _gradient_reference_colcopy(model, X, is_binary, eps):
    """Pre-optimization reference: full-matrix copy per column."""
    n, d = X.shape
    p_base = _predict(model, X, is_binary)
    grad = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        X_plus = X.copy()
        X_plus[:, j] += eps
        p_plus = _predict(model, X_plus, is_binary)
        grad[:, j] = (p_plus - p_base) / eps
    return grad


@pytest.mark.parametrize("n,d", [(200, 16), (1000, 40)])
def test_gradient_colcopy_bit_identical_and_restores_input(n, d):
    """Gradient colcopy bit identical and restores input."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    model = _LinModel(d, 1)
    eps = 0.05

    X_before = X.copy()
    ref = _gradient_reference_colcopy(model, X.copy(), is_binary=False, eps=eps)
    got = _gradient(model, X, is_binary=False, eps=eps)

    assert np.array_equal(got, ref), "optimized _gradient must be bit-identical to per-column-copy reference"
    assert np.array_equal(X, X_before), "_gradient must restore the input matrix exactly (no residual mutation)"
