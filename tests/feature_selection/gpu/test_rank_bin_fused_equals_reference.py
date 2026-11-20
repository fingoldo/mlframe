"""The fused rank-bin kernel gives the integer codes of the cupy-chain reference, ties, NaNs, constants and signed zeros included."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_rank_bin as rb


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape_bins", [(1000, 7, 10), (5003, 12, 20), (97, 6, 12), (10, 6, 20)])
def test_fused_rank_codes_equal_reference(dtype, shape_bins):
    """Every column type (tie-free, many ties, constant, NaN-holding, signed zeros) and sizes below and above the bin count give identical codes."""
    n, k, nb = shape_bins
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, k)).astype(dtype)
    X[:, 1] = np.round(X[:, 1] * 3) / 3
    X[:, 2] = 1.0
    X[rng.random(n) < 0.1, 3] = np.nan
    X[: min(5, n), 4] = -0.0
    X[min(5, n) : min(9, n), 4] = 0.0
    Xd = cp.asarray(X)
    assert bool(cp.array_equal(rb.rank_bin_codes_batch_gpu_resident(Xd, nb), rb.rank_bin_codes_batch_gpu_resident_reference(Xd, nb)))


def test_other_inputs_take_the_reference_form():
    """A one-bin request and an integer matrix go through the reference code, a one-dimensional column becomes one matrix column."""
    X = cp.asarray(np.arange(12, dtype=np.int64).reshape(6, 2))
    assert bool(cp.array_equal(rb.rank_bin_codes_batch_gpu_resident(X, 3), rb.rank_bin_codes_batch_gpu_resident_reference(X, 3)))
    col = cp.asarray(np.random.default_rng(1).standard_normal(50))
    assert rb.rank_bin_codes_batch_gpu_resident(col, 1).shape == (50, 1)
