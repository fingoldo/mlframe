"""The kept (not dispatched) device quantile codes equal the host kernel's codes."""

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,k", [(997, 12), (6000, 40)])
def test_device_codes_equal_host(dtype, n, k):
    """Continuous, tied, few-level, binary, constant and heavy-tailed columns: equal codes and code counts."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._numba_utils import quantile_codes_block
    from mlframe.feature_selection.filters._quantile_codes_gpu import quantile_codes_block_gpu

    rng = np.random.default_rng(n + k)
    v = rng.normal(size=(k, n)).astype(dtype)
    v[0] = np.round(v[0] * 2)
    v[1] = np.round(v[1] * 40)
    v[2] = 1.0
    v[3] = (rng.random(n) < 0.5).astype(dtype)
    v[4] = np.exp(v[4] * 3)
    c_h, c_d = np.empty((k, n), dtype=np.int16), np.empty((k, n), dtype=np.int16)
    nx_h, nx_d = np.empty(k, dtype=np.int64), np.empty(k, dtype=np.int64)
    quantile_codes_block(v, 10, c_h, nx_h)
    assert quantile_codes_block_gpu(v, 10, c_d, nx_d)
    np.testing.assert_array_equal(c_d, c_h)
    np.testing.assert_array_equal(nx_d, nx_h)


def test_auto_defaults_to_the_host_kernel(monkeypatch):
    """Without the override the dispatcher runs the host kernel (the device path was rejected on speed)."""
    from mlframe.feature_selection.filters import _quantile_codes_gpu as m

    monkeypatch.delenv("MLFRAME_QUANTILE_CODES_DEVICE", raising=False)
    monkeypatch.setattr(m, "quantile_codes_block_gpu", lambda *a, **k: (_ for _ in ()).throw(AssertionError("device path used")))
    v = np.random.default_rng(0).normal(size=(6, 5000)).astype(np.float32)
    codes, nx = np.empty((6, 5000), dtype=np.int16), np.empty(6, dtype=np.int64)
    m.quantile_codes_block_auto(v, 10, codes, nx)
    assert (nx > 1).all()
