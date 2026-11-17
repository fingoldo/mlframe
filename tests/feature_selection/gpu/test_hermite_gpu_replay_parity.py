"""GPU hermite helpers agree with the CPU fit/replay: double design recurrence, sigmoid slope floor, per-device gather cache."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("basis", ["hermite", "laguerre"])
def test_design_recurrence_runs_in_double_for_a_single_precision_column(basis):
    """A float32 column is widened to float64 before the recurrence, so a degree-8 design matches the double recurrence."""
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp_gpu_resident import _build_basis_matrix_gpu

    z32 = np.random.default_rng(3).normal(size=4001).astype(np.float32)
    got = _build_basis_matrix_gpu(cp, basis, cp.asarray(z32), 8)
    assert got.dtype == np.float64
    z = z32.astype(np.float64)
    want = np.empty((z.size, 9))
    want[:, 0] = 1.0
    want[:, 1] = z if basis == "hermite" else 1.0 - z
    for k in range(2, 9):
        if basis == "hermite":
            want[:, k] = z * want[:, k - 1] - (k - 1) * want[:, k - 2]
        else:
            want[:, k] = ((2 * k - 1 - z) * want[:, k - 1] - (k - 1) * want[:, k - 2]) / k
    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-10, atol=1e-10)


def test_sigmoid_slope_floor_matches_the_cpu_fit_on_a_tiny_scale_column():
    """The device sigmoid cells use the CPU fit's ``max(iqr, 1e-12)`` slope on a column whose spread is below 1e-12."""
    from mlframe.feature_selection.filters.bases import _sigmoid_fit
    from mlframe.feature_selection.filters.hermite_fe._nonpoly_cells_gpu import nonpoly_block_gpu

    x = np.random.default_rng(1).normal(size=2000) * 3e-13
    _z, params = _sigmoid_fit(x)
    block, cells = nonpoly_block_gpu(cp, cp.asarray(x.reshape(-1, 1)), "sigmoid", [5])
    assert cells == [(0, 5)]
    arg = params["slope"] * (x - float(np.quantile(x, 0.1 + 0.1 * 4)))
    want = 1.0 / (1.0 + np.exp(-arg))
    np.testing.assert_allclose(cp.asnumpy(block)[:, 0], want, rtol=1e-6, atol=1e-6)


def test_unpack_index_cache_is_keyed_by_device():
    """The packed-Gram gather indices live on one device, so the cache key carries the device id."""
    from mlframe.feature_selection.filters.hermite_fe import _als_kernels_gpu as ak

    ak._unpack_index(cp, 5)
    assert (int(cp.cuda.Device().id), 5) in ak._UNPACK
