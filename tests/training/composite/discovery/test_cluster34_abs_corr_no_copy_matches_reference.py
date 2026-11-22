"""Audit 2026-11-20 finding 34-F35: abs_corr_all_no_copy matches a float64 reference on float32 and strided inputs.

The leak filter calls this kernel for every column, on the matrix as given (no copy). The kernel must give the same correlations as
the float64 contiguous reference, so a float32 or strided ``X`` is not a silent source of different decisions.
"""

import numpy as np

from mlframe.training.composite.discovery._corr_numba import abs_corr_all_no_copy


def _reference(y, X):
    """|corr(y, X[:, j])| on a float64 contiguous copy."""
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.array([abs(np.corrcoef(yy, Xc[:, j])[0, 1]) for j in range(Xc.shape[1])])


def test_float32_strided_matrix_matches_float64_reference():
    """A strided float32 matrix gives the same absolute correlations as the float64 contiguous copy."""
    rng = np.random.default_rng(0)
    n = 400
    big = rng.normal(size=(n, 10)).astype(np.float32)
    X = big[:, ::2]  # strided view, 5 columns
    y = (X[:, 0] + 0.3 * rng.normal(size=n)).astype(np.float64)
    got = abs_corr_all_no_copy(y, X, reference_fn=_reference)
    np.testing.assert_allclose(got, _reference(y, X), rtol=1e-5, atol=1e-6)
