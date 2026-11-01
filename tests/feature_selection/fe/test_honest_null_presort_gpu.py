"""The device sort order and tie flags of the honest null equal the host's."""

import numpy as np
import pytest


def _cupy_or_skip():
    """cupy, or a skip without a usable CUDA device."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    return cp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,k", [(997, 9), (20011, 24), (50000, 12)])
def test_order_and_ties_equal_the_host(dtype, n, k):
    """Continuous, heavily tied, binary and constant columns: the same permutation (ties in the stable order) and the same flags."""
    _cupy_or_skip()
    from mlframe.feature_selection.filters._honest_null_device import _tie_flags
    from mlframe.feature_selection.filters._honest_null_presort_gpu import presort_and_ties_gpu
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(n + k)
    raw = rng.normal(size=(n, k)).astype(dtype)
    raw[:, ::3] = np.round(raw[:, ::3] * 12)
    raw[:, 1] = 2.0
    raw[:, 2] = (rng.random(n) < 0.3).astype(dtype)
    order_host = _presort_columns(raw)
    tie_host = _tie_flags(raw, order_host)
    got = presort_and_ties_gpu(raw)
    assert got is not None
    np.testing.assert_array_equal(got[0], order_host)
    np.testing.assert_array_equal(got[1], tie_host)


def test_non_finite_columns_stay_on_the_host():
    """A NaN or inf in the block makes the device path decline (the host sort handles it)."""
    _cupy_or_skip()
    from mlframe.feature_selection.filters._honest_null_presort_gpu import presort_and_ties_gpu

    raw = np.random.default_rng(0).normal(size=(100, 4)).astype(np.float32)
    raw[3, 2] = np.nan
    assert presort_and_ties_gpu(raw) is None
    raw[3, 2] = np.inf
    assert presort_and_ties_gpu(raw) is None
