"""The fused device LOF scorer equals the host scorer, for odd sizes, both dtypes, several dimensions and neighbour counts, and declines shapes it does not cover."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._lof_fe import _lof_fit_reference, _lof_transform_host
from tests._cuda_require import require_cuda_device


def _setup(n, p, ref_size, k, dtype, seed=0):
    """Rows, a reference subset and its density internals."""
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, p)) * np.linspace(1, 3, p)).astype(dtype)
    ref = X[rng.choice(n, size=ref_size, replace=False)]
    lrd, kd, k_eff = _lof_fit_reference(ref, k)
    return X, ref, lrd.astype(dtype), kd.astype(dtype), k_eff


@pytest.mark.parametrize("n", [31, 129, 5000, 20011])
@pytest.mark.parametrize("p", [1, 3, 8, 16])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_equals_host(n, p, dtype):
    """Scores agree to float rounding; the k-nearest sets are the same away from exact ties."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._lof_gpu import lof_transform_gpu

    X, ref, lrd, kd, k_eff = _setup(n, p, min(600, n - 1), 20, dtype)
    require_cuda_device()
    got = lof_transform_gpu(X, ref, lrd, kd, k_eff)
    assert got is not None and got.dtype == dtype
    want = _lof_transform_host(X, ref, lrd, kd, k_eff)
    if dtype == np.float64:
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
        return
    # float32: the host forms distances as |a|^2 + |b|^2 - 2ab in float32, the device from coordinate differences in double, so a row whose k-th and (k+1)-th neighbours are
    # almost equidistant can take the other one; those rows are rare and their scores still close
    close = np.isclose(got, want, rtol=2e-3, atol=1e-4)
    assert close.mean() >= 0.998, f"{(~close).sum()} of {n} rows differ"
    np.testing.assert_allclose(got[~close], want[~close], rtol=0.1)


def test_unsupported_shapes_fall_back():
    """More than 16 coordinates or 64 neighbours returns None, so the caller keeps the host path."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._lof_gpu import lof_transform_gpu

    X, ref, lrd, kd, k_eff = _setup(500, 17, 200, 20, np.float32)
    assert lof_transform_gpu(X, ref, lrd, kd, k_eff) is None
    X, ref, lrd, kd, _ = _setup(500, 3, 200, 20, np.float32)
    assert lof_transform_gpu(X, ref, lrd, kd, 65) is None


def test_dispatcher_follows_the_forced_path(monkeypatch):
    """_lof_transform gives the same scores on the forced host and the forced device path."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._lof_fe import _lof_transform

    X, ref, lrd, kd, k_eff = _setup(3000, 3, 500, 20, np.float32)
    monkeypatch.setenv("MLFRAME_LOF_DEVICE", "0")
    host = _lof_transform(X, ref, lrd, kd, k_eff)
    monkeypatch.setenv("MLFRAME_LOF_DEVICE", "1")
    dev = _lof_transform(X, ref, lrd, kd, k_eff)
    np.testing.assert_allclose(dev, host, rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize("k", [33, 64])
@pytest.mark.parametrize("p", [3, 16])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_equals_host_shared_topk(k, p, dtype):
    """k above the register top-k limit takes the shared-memory path; its scores match the host's."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._lof_gpu import lof_transform_gpu

    X, ref, lrd, kd, k_eff = _setup(5000, p, 600, k, dtype)
    require_cuda_device()
    got = lof_transform_gpu(X, ref, lrd, kd, k_eff)
    assert got is not None and got.dtype == dtype
    want = _lof_transform_host(X, ref, lrd, kd, k_eff)
    if dtype == np.float64:
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
        return
    close = np.isclose(got, want, rtol=2e-3, atol=1e-4)
    assert close.mean() >= 0.998, f"{(~close).sum()} of 5000 rows differ"
    np.testing.assert_allclose(got[~close], want[~close], rtol=0.1)
