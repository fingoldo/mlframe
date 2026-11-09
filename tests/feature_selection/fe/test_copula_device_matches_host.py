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
