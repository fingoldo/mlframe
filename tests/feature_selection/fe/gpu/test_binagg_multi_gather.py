"""The one-launch OOF column writer equals the per-column gather kernel for float32 and float64 blocks in both layouts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as R


@pytest.mark.parametrize("column_major", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multi_gather_equals_per_column_kernel(column_major, dtype):
    """Columns with different tables, row indices and fallbacks, NaN cells included, come out as the per-column kernel writes them."""
    rng = np.random.default_rng(0)
    n, ncols = 5000, 7
    pending = []
    for j in range(ncols):
        n_cells = 3 + j
        table = rng.standard_normal(4 * n_cells)
        table[rng.random(table.size) < 0.2] = np.nan
        idx = rng.integers(0, 4 * n_cells, n).astype(np.int64)
        pending.append((cp.asarray(idx), cp.asarray(table), float(rng.standard_normal())))
    block = cp.empty((ncols, n) if column_major else (n, ncols), dtype=dtype)
    R._oof_gather_columns_multi(cp, block, pending, column_major)
    ref = cp.empty_like(block)
    for j, (idx, table, fb) in enumerate(pending):
        R._oof_gather_into_kernel(cp, dtype)(idx, table, fb, ref[j] if column_major else ref[:, j])
    assert bool(cp.array_equal(block, ref))
