"""The device tail dependence equals the host kernel's value exactly: both are integer counts finished by the same float64 arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _tail_dependence_gpu as gpu
from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _rank_to_uniform


def _columns(n: int, dtype, seed: int):
    """A (k, n) block with ties, a constant column, columns that share the target's extremes and independent columns, plus the rank-uniformised target."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    v = np.ascontiguousarray(_rank_to_uniform(y), dtype=np.float64)
    cols = [
        y + 0.3 * rng.standard_normal(n),  # dependent everywhere
        np.where(np.abs(y) > 1.8, y, 0.0) + 0.05 * rng.standard_normal(n),  # dependent only in the extremes
        np.round(y * 2.0) / 2.0,  # heavy ties
        np.full(n, 3.0),  # constant: one run
        rng.standard_normal(n),  # independent
        np.round(rng.standard_normal(n), 1),  # independent with ties
        -y + 0.2 * rng.standard_normal(n),  # dependent, reversed
        np.where(rng.random(n) < 0.5, 0.0, -0.0),  # only signed zeros: equal values for the host, so one run
        np.where(y > 0.0, np.where(rng.random(n) < 0.5, 0.0, -0.0), rng.standard_normal(n) * 0.1),  # signed zeros mixed with a dependent tail
        np.where(y > np.quantile(y, 0.92), np.where(rng.random(n) < 0.5, 0.0, -0.0), -np.abs(rng.standard_normal(n)) - 0.01),  # the largest 8% are signed zeros: ONE run past the upper-tail rank for the host, two runs (one outside the tail) if -0.0 and 0.0 are told apart
    ]
    return np.ascontiguousarray(np.stack(cols).astype(dtype)), v


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [600, 5001, 30000])
def test_device_equals_host(dtype, n):
    """Same value per column, to the last bit, on data with ties, a constant column and real tail dependence."""
    cols, v = _columns(n, dtype, seed=n)
    want = gpu._host(cols, v)
    got = gpu.tail_dependence_columns_device(cols, v, 0.95)
    assert got is not None
    np.testing.assert_array_equal(got, want)
    assert (want > 0).any() and (want == 0).any(), "the data must exercise both a positive and a zero value, or the comparison proves little"


def test_blocks_of_one_column_give_the_same_values(monkeypatch):
    """A budget that forces one column per block changes nothing."""
    cols, v = _columns(4000, np.float32, seed=7)
    want = gpu._host(cols, v)
    monkeypatch.setattr(gpu, "_BUDGET_FRACTION", 1e-9)
    got = gpu.tail_dependence_columns_device(cols, v, 0.95)
    assert got is not None
    assert np.array_equal(got, want), (got, want)


def test_unsupported_dtype_is_left_to_the_host():
    """An integer block is not taken: the caller keeps the host kernel."""
    cols, v = _columns(500, np.float64, seed=1)
    assert gpu.tail_dependence_columns_device(cols.astype(np.int64), v, 0.95) is None


@pytest.mark.parametrize("n", [700, 12000])
def test_packed_kernel_equals_the_chain_version_on_float32(n):
    """The fused-kernel version and the reference chain version give the same four counts, including for signed zeros and ties."""
    import cupy as cp

    cols, v = _columns(n, np.float32, seed=n + 3)
    hi, lo = 0.95, 1.0 - 0.95
    vflag = (cp.asarray(v > hi).astype(cp.uint64) << cp.uint64(1)) | cp.asarray(v < lo).astype(cp.uint64)
    x = cp.asarray(cols)
    packed = gpu._block_counts_packed(cp, x, vflag, n, hi, lo)
    chain = gpu._block_counts_chain(cp, x, cp.asarray(v), n, hi, lo)
    np.testing.assert_array_equal(packed, chain)
