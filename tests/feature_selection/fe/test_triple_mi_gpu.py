"""The device triple MI equals the host kernel, alone and through the triplet enumeration."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_triplet_fe as T


def _gpu_or_skip():
    """The device module, or a skip without a usable CUDA device."""
    cp = pytest.importorskip("cupy")
    try:  # audit-exempt: except_skip_masks_call_under_test -- the guarded call only asks the runtime for a CUDA device; nothing under test runs in the try
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters import _triple_mi_gpu

    return _triple_mi_gpu


@pytest.mark.parametrize("n,p,n_y,base", [(997, 6, 2, 5), (4001, 9, 3, 5), (12000, 13, 4, 4), (3000, 5, 2, 6)])
def test_scores_equal_the_host_kernel(n, p, n_y, base):
    """Odd sizes, 2 to 4 classes and different level counts: same triples in the same order, same MI."""
    gpu = _gpu_or_skip()
    rng = np.random.default_rng(n + p)
    codes = rng.integers(0, base, size=(p, n)).astype(np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    codes[0] = (codes[0] + y) % base
    got = gpu.triple_mi_gpu(codes, y, base, n_y)
    want = T._triple_mi_kernel(codes, y, base, n_y)
    assert got is not None
    assert list(zip(got[:3], want[:3]))
    for a, b in zip(got[:3], want[:3]):
        assert np.array_equal(a, b, equal_nan=True)
    assert np.allclose(got[3], want[3], rtol=0, atol=1e-12, equal_nan=True)


def test_oversized_table_falls_back():
    """A table above the shared budget and a target with more than 16 classes are declined (the host kernel then runs)."""
    gpu = _gpu_or_skip()
    codes = np.zeros((4, 100), dtype=np.int64)
    assert gpu.triple_mi_gpu(codes, np.zeros(100, dtype=np.int64), 5, 17) is None  # more than 16 classes
    assert gpu.triple_mi_gpu(codes, np.zeros(100, dtype=np.int64), 20, 2) is None


def test_enumeration_is_the_same_on_both_paths(monkeypatch):
    """The ranked triples do not depend on the path."""
    _gpu_or_skip()
    rng = np.random.default_rng(8)
    n = 6000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(10)})
    y = ((X["x1"] * X["x2"] * X["x3"]) > 0).astype(int).to_numpy()
    monkeypatch.setenv("MLFRAME_TRIPLE_MI_DEVICE", "0")
    host = T._enumerate_triplet_candidates(X, list(X.columns), y)
    monkeypatch.setenv("MLFRAME_TRIPLE_MI_DEVICE", "1")
    dev = T._enumerate_triplet_candidates(X, list(X.columns), y)
    assert host == dev and host[0] == ("x1", "x2", "x3")
