"""The device quantile codes (radix selection for float32, the sort variant otherwise) equal the host kernel's codes."""

import numpy as np
import pytest
from tests._cuda_require import require_cuda_device


@pytest.mark.parametrize("dtype,method", [(np.float32, "radix"), (np.float32, "sort"), (np.float64, "radix")])
@pytest.mark.parametrize("n,k", [(997, 12), (6000, 40), (20011, 9)])
def test_device_codes_equal_host(dtype, method, n, k):
    """Continuous, tied, few-level, binary, constant and heavy-tailed columns: equal codes and code counts."""
    pytest.importorskip("cupy")
    require_cuda_device()
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
    assert quantile_codes_block_gpu(v, 10, c_d, nx_d, method=method)
    np.testing.assert_array_equal(c_d, c_h)
    np.testing.assert_array_equal(nx_d, nx_h)


def test_auto_dispatch_follows_size_dtype_and_override(monkeypatch):
    """Small blocks and float64 blocks stay on the host kernel, a large float32 block (or the override) goes to the device, a device fault returns to the host."""
    from mlframe.feature_selection.filters import _quantile_codes_gpu as m

    calls = []
    monkeypatch.setattr(m, "quantile_codes_block_gpu", lambda *a, **k: calls.append("device") or True)
    host = lambda *a, **k: calls.append("host")
    small = np.zeros((6, 5000), dtype=np.float32)
    codes, nx = np.empty((6, 5000), dtype=np.int16), np.empty(6, dtype=np.int64)
    monkeypatch.delenv("MLFRAME_QUANTILE_CODES_DEVICE", raising=False)
    m.quantile_codes_block_auto(small, 10, codes, nx, host_kernel=host)
    assert calls == ["host"]
    calls.clear()
    monkeypatch.setenv("MLFRAME_QUANTILE_CODES_DEVICE", "1")
    m.quantile_codes_block_auto(small, 10, codes, nx, host_kernel=host)
    assert calls == ["device"]
    calls.clear()
    monkeypatch.setattr(m, "quantile_codes_block_gpu", lambda *a, **k: (_ for _ in ()).throw(MemoryError("pool")))
    m.quantile_codes_block_auto(small, 10, codes, nx, host_kernel=host)
    assert calls == ["host"]
    calls.clear()
    monkeypatch.delenv("MLFRAME_QUANTILE_CODES_DEVICE")
    monkeypatch.setattr(m, "DEVICE_MIN_ELEMENTS", 1000)
    monkeypatch.setattr(m, "quantile_codes_block_gpu", lambda *a, **k: calls.append("device") or True)
    m.quantile_codes_block_auto(small, 10, codes, nx, host_kernel=host)
    m.quantile_codes_block_auto(small.astype(np.float64), 10, codes, nx, host_kernel=host)
    assert calls == ["device", "host"]
