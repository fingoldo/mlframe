"""The fused observed-MI kernel sized from a known code bound returns the same MI and occupied bins as sizing it from the codes' maximum (MRMR-SYNC, 2026-11-18)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_analytic_gpu import resident_observed_mi_and_bins


@pytest.mark.parametrize("n_rows, n_cols", [(4000, 25), (20000, 60)])
def test_code_bound_gives_the_same_mi_and_bins_as_the_code_maximum(n_rows: int, n_cols: int):
    """Codes in [0, 7] with a declared bound of 10: the bins are exact and the MI agrees to 1e-12 (the launch shape changes, the cells that count do not)."""
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 8, size=(n_rows, n_cols)).astype(np.int8)
    codes[:, 3] = 0  # a constant column: one occupied bin
    y = (rng.random(n_rows) < 0.4).astype(np.int64)
    dev = cp.asarray(codes)
    mi_max, bins_max = resident_observed_mi_and_bins(dev, y, 2)
    mi_bound, bins_bound = resident_observed_mi_and_bins(dev, y, 2, code_bound=10)
    np.testing.assert_array_equal(bins_bound, bins_max)
    np.testing.assert_allclose(mi_bound, mi_max, rtol=0, atol=1e-12)
    assert bins_bound[3] == 1
