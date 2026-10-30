"""``scrub_nonfinite`` zeroes NaN and infinities in place exactly as ``np.nan_to_num(nan=0, posinf=0, neginf=0)`` does."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_code_bins import scrub_nonfinite


def _dirty(shape, dtype, seed=0):
    """Random values with NaN, +inf, -inf and the largest finite values sprinkled in."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(dtype)
    flat = a.reshape(-1)
    idx = rng.choice(flat.size, 12, replace=False)
    flat[idx[:3]], flat[idx[3:6]], flat[idx[6:9]] = np.nan, np.inf, -np.inf
    flat[idx[9:]] = [np.finfo(dtype).max, -np.finfo(dtype).max, np.finfo(dtype).tiny]
    return a


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(500,), (50, 7)])
def test_matches_nan_to_num(dtype, shape):
    """Same result for 1-D and 2-D float arrays, contiguous or strided."""
    a = _dirty(shape, dtype)
    want = np.nan_to_num(a.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    got = a.copy()
    scrub_nonfinite(got)
    assert np.array_equal(got, want)
    wide = _dirty((shape[0], 4), dtype, seed=1)
    view = wide[:, 1]
    want_col = np.nan_to_num(view.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    scrub_nonfinite(view)
    assert np.array_equal(view, want_col)


def test_integer_arrays_are_left_to_numpy():
    """A non-float array is handled by ``np.nan_to_num`` and is unchanged."""
    a = np.arange(10, dtype=np.int32)
    scrub_nonfinite(a)
    assert np.array_equal(a, np.arange(10))
