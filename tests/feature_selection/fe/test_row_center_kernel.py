"""The one-pass row centring kernel gives the numpy chain's result."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup import _row_center_finite


def _reference(A):
    """The numpy chain the kernel replaced."""
    finite = np.isfinite(A)
    counts = finite.sum(axis=1, keepdims=True)
    sums = np.where(finite, np.asarray(A, dtype=np.float64), 0.0).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.asarray(np.asarray(A, dtype=np.float64) - means, dtype=A.dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kernel_matches_numpy_chain(dtype):
    """Finite, NaN, inf and all-NaN rows, an offset far above the spread, and odd widths."""
    rng = np.random.default_rng(0)
    A = (rng.normal(size=(37, 1001)) * 0.05 + 8.5e3).astype(dtype)
    A[3, ::7] = np.nan
    A[5, 10] = np.inf
    A[8, :] = np.nan
    got, want = _row_center_finite(A), _reference(A)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=0, atol=5e-5 if dtype == np.float32 else 1e-9, equal_nan=True)
    assert np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(np.isinf(got), np.isinf(want))
