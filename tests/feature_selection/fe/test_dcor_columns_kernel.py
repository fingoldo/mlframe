"""The one-pass dCor kernel gives the dense double-centred matrix values."""

import numpy as np

from mlframe.feature_selection.filters import _orthogonal_dcor_fe as D


def _dense_dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely-Rizzo distance correlation in float64 from the full double-centred matrices."""
    def centred(z):
        """Double-centred absolute-difference matrix of ``z``."""
        a = np.abs(z[:, None] - z[None, :])
        return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()

    A, B = centred(x.astype(np.float64)), centred(y.astype(np.float64))
    dcov2 = max(float(np.mean(A * B)), 0.0)
    return float(np.sqrt(dcov2 / np.sqrt(float(np.mean(A * A)) * float(np.mean(B * B)))))


def test_kernel_matches_dense_values_and_guards_constant_columns():
    """Continuous, tied and constant columns, on a subsample smaller than the data."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    X[:, 1] = np.round(X[:, 1])
    X[:, 2] = 1.0
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    got = D._dcor_batch(X, y, n_sample=400, random_state=0)
    idx = D._subsample_indices(n, 400, 0)
    for j in (0, 3, 4):
        np.testing.assert_allclose(got[j], _dense_dcor(X[idx, j], y[idx]), rtol=2e-4, atol=1e-6)
    assert got[2] == 0.0
