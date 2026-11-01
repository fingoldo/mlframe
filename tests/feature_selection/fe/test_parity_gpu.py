"""The device parity grid scan equals the host njit scan, alone and through ``hybrid_parity_fe_with_recipes``."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _parity_fe as P


def _device_or_skip():
    """The device module, or a skip without a usable CUDA device."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters import _parity_gpu

    return _parity_gpu


@pytest.mark.parametrize("n,k,nc", [(997, 5, 2), (5003, 9, 3), (20000, 33, 4), (4001, 40, 2)])
def test_grid_mi_equals_the_host_kernel(n, k, nc):
    """Odd sizes, tied columns, 2 to 4 classes, more columns than one launch block: the same MI per period."""
    gpu = _device_or_skip()
    rng = np.random.default_rng(n + k)
    cols = [rng.normal(size=n) * s + off for s, off in zip(rng.uniform(0.5, 50, k), rng.uniform(-100, 100, k))]
    cols[1] = np.round(cols[1], 1)
    y = rng.integers(0, nc, n)
    yc = P._dense_codes(y)
    grids = [P._parity_grid(c) for c in cols]
    got = gpu.parity_mi_grids_gpu(cols, yc, nc, grids)
    assert got is not None and len(got) == k
    for c, g, mi in zip(cols, grids, got):
        want = P._parity_mi_grid_njit(np.ascontiguousarray(c), yc, nc, g)
        np.testing.assert_allclose(mi, want, rtol=0, atol=1e-12)
        assert int(np.argmax(mi)) == int(np.argmax(want))


def test_more_than_four_classes_stay_on_the_host():
    """Five classes are outside the kernel: the helper declines."""
    gpu = _device_or_skip()
    cols = [np.random.default_rng(0).normal(size=500)]
    assert gpu.parity_mi_grids_gpu(cols, np.arange(500) % 5, 5, [P._parity_grid(cols[0])]) is None


def test_hybrid_parity_device_equals_host(monkeypatch):
    """The scores frame, the appended names and the recipes do not depend on the path (a NaN column keeps the host scan)."""
    _device_or_skip()
    rng = np.random.default_rng(3)
    n = 6000
    X = pd.DataFrame({f"x{i}": rng.uniform(0, 20, n) for i in range(8)})
    X.loc[::17, "x4"] = np.nan
    y = (np.floor(X["x1"].to_numpy() / 2.5).astype(int) % 2).astype(np.int64)
    monkeypatch.setenv("MLFRAME_PARITY_DEVICE", "0")
    host = P.hybrid_parity_fe_with_recipes(X, y)
    monkeypatch.setenv("MLFRAME_PARITY_DEVICE", "1")
    dev = P.hybrid_parity_fe_with_recipes(X, y)
    pd.testing.assert_frame_equal(host[3], dev[3], check_exact=False, atol=1e-12, rtol=0)
    assert host[1] == dev[1] and len(host[1]) > 0
    pd.testing.assert_frame_equal(host[0], dev[0])
