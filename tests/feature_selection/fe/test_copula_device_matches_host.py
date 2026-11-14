"""The device copula MI gives the host kernel's value per column, with continuous, heavily tied, constant and signed-zero columns, at float32 and float64."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_copula_gpu as gpu
from mlframe.feature_selection.filters import _orthogonal_copula_mi_fe as host


def _inputs(n, k, seed, dtype):
    """Columns of several kinds and the 20 bin codes of a target's ranks."""
    r = np.random.default_rng(seed)
    y = r.standard_normal(n)
    cols = r.standard_normal((k, n))
    cols[0] += y  # informative
    cols[1] = np.round(cols[1] * 2)  # ties
    cols[2] = r.integers(0, 4, n)  # four values
    cols[3] = 1.5  # constant
    cols[4] = np.where(r.random(n) < 0.5, 0.0, -0.0)  # signed zeros
    cols = np.ascontiguousarray(cols.astype(dtype))
    ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable") + 1.0
    vi = np.clip(np.floor(ranks / (n + 1.0) * 20).astype(np.int64), 0, 19)
    return cols, vi


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,k", [(1500, 7), (20000, 9), (40001, 6)])
def test_device_copula_equals_host(n, k, dtype):
    """Per-column copula MI equals the host kernel's (integer joint tables, the same final sum)."""
    cols, vi = _inputs(n, k, 0, dtype)
    want = np.empty(k)
    (host._copula_columns_radix_njit if dtype == np.float32 else host._copula_columns_njit)(cols, vi, 20, want)
    got = gpu.copula_columns_device(cols, vi, 20)
    assert got is not None
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert want[0] > want[3]  # the informative column scores above the constant one


def test_public_batch_routes_through_the_device_when_forced(monkeypatch):
    """``_copula_mi_batch`` with the device chosen equals the host path chosen off."""
    r = np.random.default_rng(3)
    n = 6000
    y = r.standard_normal(n)
    X = r.standard_normal((n, 8)).astype(np.float32)
    X[:, 0] += y
    X[:, 1] = np.round(X[:, 1])
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    want = host._copula_mi_batch(X, y, n_bins=20)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    got = host._copula_mi_batch(X, y, n_bins=20)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_resident_block_is_used_by_the_batch_when_the_device_is_chosen(monkeypatch):
    """A caller that already holds the columns on the device gets the same values through ``device_block`` as through the host path."""
    r = np.random.default_rng(4)
    n = 5000
    y = r.standard_normal(n)
    X = r.standard_normal((n, 6)).astype(np.float32)
    X[:, 2] = np.round(X[:, 2])
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    want = host._copula_mi_batch(X, y, n_bins=20)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    got = host._copula_mi_batch(X, y, n_bins=20, device_block=cp.asarray(np.ascontiguousarray(X.T)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_copula_equals_host_on_independent_seeds(seed, dtype):
    """The device copula MI equals the host kernel on further independent draws, not only the one seed the size grid uses."""
    cols, vi = _inputs(12000, 8, seed, dtype)
    want = np.empty(8)
    (host._copula_columns_radix_njit if dtype == np.float32 else host._copula_columns_njit)(cols, vi, 20, want)
    got = gpu.copula_columns_device(cols, vi, 20)
    assert got is not None
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (got, want)


@pytest.mark.parametrize("free_bytes", [1 << 22, 1 << 24])
def test_streamed_blocks_equal_the_unblocked_result(monkeypatch, free_bytes):
    """A small free-memory figure forces several column blocks; the streamed result equals the one-block result exactly."""
    cols, vi = _inputs(20000, 9, 0, np.float32)
    whole = gpu.copula_columns_device(cols, vi, 20)
    monkeypatch.setattr(gpu, "device_room", lambda cp: free_bytes)
    streamed = gpu.copula_columns_device(cols, vi, 20)
    assert whole is not None and streamed is not None
    np.testing.assert_array_equal(streamed, whole)


def test_device_fault_falls_back_to_none(monkeypatch):
    """A failing device call returns None (the caller runs the host kernel) instead of ending the request."""
    cols, vi = _inputs(2000, 6, 0, np.float32)

    def broken(cp):
        """Stand-in for a device that cannot report its free memory."""
        raise RuntimeError("no device memory figure")

    monkeypatch.setattr(gpu, "device_room", broken)
    assert gpu.copula_columns_device(cols, vi, 20) is None
