"""The device Xi gives the host kernel's value per column, with continuous, heavily tied and constant columns, at float32 and float64."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_xi_fe as host
from mlframe.feature_selection.filters import _orthogonal_xi_gpu as gpu


def _inputs(n, k, seed, dtype):
    """Columns of several kinds, their permutations and the target ranks of a 3-class target."""
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n).astype(np.float64)
    cols = r.standard_normal((k, n))
    cols[0] += y  # informative
    cols[1] = np.round(cols[1] * 2)  # ties
    cols[2] = r.integers(0, 4, n)  # four values
    cols[3] = 1.5  # constant
    cols[4] = np.where(r.random(n) < 0.5, 0.0, -0.0)  # signed zeros
    cols = np.ascontiguousarray(cols.astype(dtype))
    perms = np.stack([r.permutation(n) for _ in range(k)])
    _u, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    r_row = np.cumsum(counts)[inv].astype(np.float64)
    l_row = np.cumsum(counts[::-1])[::-1][inv].astype(np.float64)
    return cols, perms, r_row, l_row


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,k", [(1500, 7), (20000, 9), (40001, 6)])
def test_device_xi_equals_host(n, k, dtype):
    """Per-column Xi is identical to the host kernel (integer rank sums, same tie-break permutations)."""
    cols, perms, r_row, l_row = _inputs(n, k, 0, dtype)
    want = np.empty(k)
    kernel = host._xi_columns_radix_njit if dtype == np.float32 else host._xi_columns_njit
    kernel(cols, perms, r_row, l_row, want)
    got = gpu.xi_columns_device(cols, perms, r_row, l_row)
    assert got is not None
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert want[0] > want[3]  # the informative column scores above the constant one


def test_public_batch_routes_through_the_device_when_forced(monkeypatch):
    """``xi_correlation_batch`` with the device chosen equals the host path chosen off."""
    r = np.random.default_rng(3)
    n = 6000
    y = r.integers(0, 3, n)
    X = r.standard_normal((n, 8)).astype(np.float32)
    X[:, 0] += y
    X[:, 1] = np.round(X[:, 1])
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    want = host.xi_correlation_batch(X, y, random_state=5)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    got = host.xi_correlation_batch(X, y, random_state=5)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
