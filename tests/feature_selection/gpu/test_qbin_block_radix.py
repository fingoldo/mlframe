"""The block binner's order statistics from the radix select give the codes of the row sort, for every row shape."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._qbin_block_gpu import sync_free_qbin_codes_block


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,nbins", [(50000, 10), (30001, 16), (997, 4), (8, 10), (1, 5), (12345, 63)])
def test_codes_equal_the_sort_path(monkeypatch, dtype, n, nbins):
    """Normal, constant, two-valued, tied, heavy-tailed and tiny-range rows."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((7, n)).astype(dtype)
    x[1] = 2.5
    x[2] = (rng.random(n) < 0.3).astype(dtype)
    x[3] = np.round(x[3] * 2)
    x[4] = rng.standard_cauchy(n).astype(dtype)
    x[5] = (1.0 + 1e-6 * rng.random(n)).astype(dtype)
    xd = cp.asarray(x)
    monkeypatch.setenv("MLFRAME_QBIN_BLOCK_RADIX", "0")
    want = sync_free_qbin_codes_block(cp, xd, nbins)
    monkeypatch.setenv("MLFRAME_QBIN_BLOCK_RADIX", "1")
    got = sync_free_qbin_codes_block(cp, xd, nbins)
    assert cp.array_equal(got, want)
