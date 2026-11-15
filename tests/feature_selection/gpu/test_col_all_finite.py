"""The coalesced column finiteness kernel of the cupy polynom optimizer equals ``isfinite(x).all(axis=0)`` for float32 and double, with NaN/inf in first, middle and last rows and ragged slab sizes."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _cupy_polynom_optimizer as m


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,p", [(1, 3), (63, 5), (64, 100), (1000, 33), (100_003, 100)])
def test_matches_isfinite_all(dtype, n, p):
    """Bad cells at the first row, the last row and in the middle land in the right columns; clean columns stay True."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, p)).astype(dtype)
    x[0, 0] = np.nan
    x[n - 1, p - 1] = np.inf
    if n > 2 and p > 2:
        x[n // 2, p // 2] = -np.inf
    got = cp.asnumpy(m._col_all_finite(cp, cp.asarray(x)))
    np.testing.assert_array_equal(got, np.isfinite(x).all(axis=0))
