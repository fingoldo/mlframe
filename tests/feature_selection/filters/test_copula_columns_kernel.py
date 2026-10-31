"""The parallel copula-MI kernel agrees with the per-column rank + joint-histogram path on ties, constants, NaN columns and a continuous target."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _bin_mi_uniform_pair, _copula_mi_batch, _rank_to_uniform


def _per_column(X: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """The scalar path: rank-transform both sides over the jointly finite rows and bin."""
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j]) & np.isfinite(y)
        if ok.sum() >= 2:
            out[j] = _bin_mi_uniform_pair(_rank_to_uniform(X[ok, j]), _rank_to_uniform(y[ok]), n_bins=n_bins)
    return out


def test_kernel_matches_scalar_path():
    """Continuous, rounded, constant, binary and NaN-bearing columns against a discrete and a continuous target."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 8))
    X[:, 1] = np.round(X[:, 1], 1)
    X[:, 2] = 1.0
    X[:, 3] = (X[:, 3] > 0).astype(float)
    X[11, 4] = np.nan
    for y in ((X[:, 0] ** 2 + rng.normal(size=n) > 1).astype(np.int64), X[:, 0] * X[:, 5] + rng.normal(size=n)):
        got = _copula_mi_batch(X, y, n_bins=20)
        ref = _per_column(X, np.asarray(y, dtype=np.float64), 20)
        np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3)
        assert np.argmax(got) == np.argmax(ref)
