"""The fused per-cell moment kernels agree with the scatter-add form and fall back when they do not apply."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as res
from mlframe.feature_selection.filters import _cell_moments_kernel as kern


def _inputs(n, n_cells, seed=0, zero_frac=0.2):
    """Cell codes, finite values with an offset and a 0/1 row weight."""
    rng = np.random.default_rng(seed)
    codes = cp.asarray(rng.integers(0, n_cells, size=n).astype(np.int64))
    v = cp.asarray(rng.normal(1000.0, 3.0, size=n))
    w = cp.asarray((rng.random(n) > zero_frac).astype(np.float64))
    return codes, v, w


@pytest.mark.parametrize("n,n_cells", [(200_000, 12), (50_000, 300), (10, 3), (1_000, 1)])
def test_fused_moments_match_the_scatter_form(n, n_cells):
    """Count and mean exactly (up to summation order), the centred moments to 1e-7 (both forms add with atomics in an unspecified order, so cancelling sums differ at summation-order noise), including cells that the weights leave empty."""
    codes, v, w = _inputs(n, n_cells)
    got = kern.masked_cell_moments(cp, codes, v, w, n_cells)
    assert got is not None
    want = res._per_cell_moments_stable_masked_scatter_gpu(cp, codes, v, w, n_cells)
    for g, e in zip(got, want):
        np.testing.assert_allclose(cp.asnumpy(g), cp.asnumpy(e), rtol=1e-7, atol=1e-7)


def test_too_many_cells_for_shared_memory_declines():
    """Above MAX_CELLS the kernel steps aside (None) and the dispatcher uses the scatter form."""
    codes, v, w = _inputs(1_000, 5)
    assert kern.masked_cell_moments(cp, codes, v, w, kern.MAX_CELLS + 1) is None
    assert len(res._per_cell_moments_stable_masked_gpu(cp, codes, v, w, 5)) == 5


def test_an_all_zero_weight_gives_empty_cells():
    """No row counts: every cell is empty (count 0), as the degenerate-fold case requires."""
    codes, v, _ = _inputs(1_000, 4)
    w = cp.zeros(1_000)
    cnt, mean, cm2, cm3, cm4 = kern.masked_cell_moments(cp, codes, v, w, 4)
    assert float(cm3.sum()) == 0.0 and float(cm4.sum()) == 0.0
    assert float(cnt.sum()) == 0.0 and float(cm2.sum()) == 0.0 and float(mean.sum()) == 0.0
