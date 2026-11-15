"""The fused ``bf_apply`` kernel returns, for every binary function, exactly the values of the cupy-op chain it replaced and flags the columns whose inputs or result are non-finite (float32 and double)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _cupy_polynom_optimizer as m


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bf", list(m._BF_ORDER))
@pytest.mark.parametrize("n,p", [(1, 3), (257, 5), (30_001, 100)])
def test_matches_the_cupy_chain(dtype, bf, n, p):
    """Values equal (NaN in the same cells), the finite mask equals isfinite().all(axis=0) of inputs and result; exact zeros in B cover the safe divide, tiny values the logabs sign."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    rng = np.random.default_rng(7)
    a = rng.standard_normal((n, p)).astype(dtype)
    b = rng.standard_normal((n, p)).astype(dtype)
    b[rng.random((n, p)) < 0.01] = 0.0
    a[rng.random((n, p)) < 0.01] = 0.0
    a[0, 0] = np.nan
    if p > 2:
        b[n - 1, p - 1] = np.inf
    ad, bd = cp.asarray(a), cp.asarray(b)
    got, ok = m._bf_apply_checked(cp, m._BF_ORDER.index(bf), ad, bd)
    with np.errstate(all="ignore"):
        want = cp.asnumpy(m._bf_apply_gpu(cp, bf, ad, bd))
    got_h = cp.asnumpy(got)
    np.testing.assert_array_equal(np.isnan(got_h), np.isnan(want))
    np.testing.assert_array_equal(got_h[~np.isnan(want)], want[~np.isnan(want)])
    expected_ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0) & np.isfinite(want).all(axis=0)
    np.testing.assert_array_equal(cp.asnumpy(ok), expected_ok)
