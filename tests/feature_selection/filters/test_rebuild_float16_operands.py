"""Float16 operand columns are promoted to float32 before the numba transforms of the full-sample survivor rebuild (numba has no float16)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.feature_engineering import _numba_float


def test_float16_promoted_exactly():
    """Same values, float32 dtype; other dtypes pass through untouched."""
    x16 = (np.random.default_rng(0).standard_normal(1000) * 10).astype(np.float16)
    out = _numba_float(x16)
    assert out.dtype == np.float32
    assert np.array_equal(out, x16.astype(np.float64))
    x32 = x16.astype(np.float32)
    assert _numba_float(x32) is x32
    xi = np.arange(5, dtype=np.int8)
    assert _numba_float(xi) is xi
