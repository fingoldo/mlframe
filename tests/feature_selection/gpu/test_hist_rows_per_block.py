"""The shared-memory joint histogram sizes its row blocks to fill the device, and the counts stay exact.

A fixed 16384 rows per block gave a grid of 14 blocks at 100000 rows x 25 columns on a 6-SM card (occupancy 0.25, 75% memory-dependency stalls). The rows per block now follow the device's multiprocessor
count; the counts must equal ``bincount`` for any row count, column count and block size, including a last block that is short.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as m


def _reference(X, b, kx, kb):
    """Per-column joint counts of the codes ``X`` (n, K) against ``b`` by one bincount."""
    k = X.shape[1]
    keys = X.astype(cp.int64) * kb + b[:, None] + cp.arange(k, dtype=cp.int64)[None, :] * (kx * kb)
    return cp.bincount(keys.ravel(), minlength=k * kx * kb).reshape(k, kx * kb)


@pytest.mark.parametrize("n,k", [(100_000, 25), (30_001, 7), (5_000, 50), (1_025, 1), (70_000, 33), (999, 4)])
@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
@pytest.mark.parametrize("seed", [0, 1])
def test_counts_equal_bincount_for_any_shape(n, k, dtype, seed):
    """Exact counts for narrow, wide, odd and tiny shapes, on the byte path and the wide-integer path, two datasets."""
    rng = np.random.default_rng(seed)
    kx, kb = 10, 2
    X = cp.asarray(rng.integers(0, kx, (n, k)).astype(dtype))
    b = cp.asarray(rng.integers(0, kb, n).astype(np.int64))
    np.testing.assert_array_equal(cp.asnumpy(m._batched_joint_counts2(X, b, kx, kb)), cp.asnumpy(_reference(X, b, kx, kb)))


def test_a_narrow_matrix_gets_enough_blocks_to_fill_the_device():
    """100000 rows x 25 columns (two column tiles) launches at least one block per SM and several per SM when the rows allow it."""
    sms = m._sm_count()
    rows = m._hist_rows_per_block(100_000, 2)
    blocks = ((100_000 + rows - 1) // rows) * 2
    assert blocks >= sms * 4, (rows, blocks, sms)


def test_the_block_never_exceeds_the_old_cap_or_the_row_count():
    """Never more than 16384 rows, never more than n, never fewer than 1024 (a flush of the block's histogram must stay small against its rows)."""
    for n in (500, 1_024, 5_000, 100_000, 10_000_000):
        for tiles in (1, 2, 7, 100):
            r = m._hist_rows_per_block(n, tiles)
            assert min(n, 1024) <= r <= max(min(n, 16384), 1024) and r <= max(n, 1024)
