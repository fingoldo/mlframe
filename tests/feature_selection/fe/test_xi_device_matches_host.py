"""The device Xi gives the host kernel's value per column, with continuous, heavily tied, constant and signed-zero columns, from host or resident operands."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_xi_fe as host
from mlframe.feature_selection.filters import _orthogonal_xi_gpu as gpu


def _inputs(n, k, seed):
    """float32 columns of several kinds, their tie seeds and the target ranks of a 3-class target."""
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n).astype(np.float64)
    cols = r.standard_normal((k, n))
    cols[0] += y  # informative
    cols[1] = np.round(cols[1] * 2)  # ties
    cols[2] = r.integers(0, 4, n)  # four values
    cols[3] = 1.5  # constant
    cols[4] = np.where(r.random(n) < 0.5, 0.0, -0.0)  # signed zeros
    cols = np.ascontiguousarray(cols.astype(np.float32))
    r_row, l_row = host._target_ranks(y)
    return cols, host.column_seeds(seed, k), r_row, l_row


@pytest.mark.parametrize("n,k", [(1500, 7), (20000, 9), (40001, 6)])
@pytest.mark.parametrize("resident", [False, True])
def test_device_xi_equals_host(n, k, resident):
    """Per-column Xi is identical to the host kernel (integer rank sums, the same counter-hash tie keys), whether the columns are uploaded or already on the device."""
    cols, seeds, r_row, l_row = _inputs(n, k, 0)
    want = np.empty(k)
    host._xi_columns_radix_njit(cols, seeds, r_row, l_row, want)
    got = gpu.xi_columns_device(cp.asarray(cols) if resident else cols, seeds, r_row, l_row)
    assert got is not None
    np.testing.assert_array_equal(got, want)
    assert want[0] > want[3]  # the informative column scores above the constant one


@pytest.mark.parametrize("n,k", [(5000, 9), (30000, 6)])
@pytest.mark.parametrize("resident", [False, True])
def test_float64_device_xi_equals_host(n, k, resident):
    """Float64 columns on the device give the host float64 kernel's Xi per column: ties, a constant column and signed zeros included."""
    cols, seeds, r_row, l_row = _inputs(n, k, 1)
    wide = np.ascontiguousarray(cols.astype(np.float64))
    want = np.empty(k)
    host._xi_columns_njit(wide, seeds, r_row, l_row, want)
    got = gpu.xi_columns_device(cp.asarray(wide) if resident else wide, seeds, r_row, l_row, allow_wide=True)
    assert got is not None
    assert np.array_equal(got, want), (got, want)


def test_public_batch_routes_through_the_device_when_chosen(monkeypatch):
    """``xi_correlation_batch`` with the device chosen equals the host path."""
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
    np.testing.assert_array_equal(got, want)


def test_resident_block_is_used_by_the_batch_when_the_device_is_chosen(monkeypatch):
    """A caller that already holds the columns on the device gets the same values through ``device_block`` as through the host path."""
    r = np.random.default_rng(4)
    n = 5000
    y = r.integers(0, 2, n)
    X = r.standard_normal((n, 6)).astype(np.float32)
    X[:, 2] = np.round(X[:, 2])
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    want = host.xi_correlation_batch(X, y, random_state=9)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    got = host.xi_correlation_batch(X, y, random_state=9, device_block=cp.asarray(np.ascontiguousarray(X.T)))
    np.testing.assert_array_equal(got, want)


def test_float64_stays_on_the_host_by_default():
    """The production criterion dtype is float32; a float64 block without the opt-in is handed back to the host kernel."""
    cols, seeds, r_row, l_row = _inputs(3000, 5, 2)
    assert gpu.xi_columns_device(cols.astype(np.float64), seeds, r_row, l_row) is None


@pytest.mark.parametrize("free_bytes", [1 << 22, 1 << 24])
def test_streamed_blocks_equal_the_unblocked_result(monkeypatch, free_bytes):
    """A small free-memory figure forces several column blocks; the streamed float32 result equals the one-block result exactly."""
    cols, seeds, r_row, l_row = _inputs(20000, 9, 0)
    whole = gpu.xi_columns_device(cols, seeds, r_row, l_row)
    monkeypatch.setattr(gpu, "device_room", lambda cp: free_bytes)
    streamed = gpu.xi_columns_device(cols, seeds, r_row, l_row)
    assert whole is not None and streamed is not None
    np.testing.assert_array_equal(streamed, whole)


def test_device_fault_falls_back_to_none(monkeypatch):
    """A failing device call returns None so the caller runs the host kernel, and the request is not ended."""
    cols, seeds, r_row, l_row = _inputs(2000, 6, 0)

    def broken(cp):
        """Stand-in for a device that cannot report its free memory."""
        raise RuntimeError("no device memory figure")

    monkeypatch.setattr(gpu, "device_room", broken)
    assert gpu.xi_columns_device(cols, seeds, r_row, l_row) is None
