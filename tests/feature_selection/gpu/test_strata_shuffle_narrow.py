"""The byte-matrix strata shuffle gives the values of the int64 shuffle for the same generator state (MRMR-STRATA-U8, 2026-11-19)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

import mlframe.feature_selection.filters._fe_cmi_perm_null_gpu as G
from mlframe.feature_selection.filters._strata_shuffle_gpu import strata_shuffle


@pytest.mark.parametrize("n, nperm, nrank, kx", [(30000, 25, 4000, 10), (100000, 25, 14880, 10), (6000, 40, 3500, 2)])
def test_narrow_equals_wide_and_keeps_strata_values(n: int, nperm: int, nrank: int, kx: int):
    """Same seed: the uint8 result equals the int64 result element for element, and every stratum keeps its values."""
    rng = np.random.default_rng(4)
    z = np.sort(rng.integers(0, nrank, n))
    _, z = np.unique(z, return_inverse=True)
    x = rng.integers(0, kx, n).astype(np.int64)
    nr = int(z.max()) + 1
    zd, xd = cp.asarray(z.astype(np.float64)), cp.asarray(x)
    wide = strata_shuffle(cp, G._DeviceRng(cp, 11), xd, zd, nperm, nr, narrow=False)
    narrow = strata_shuffle(cp, G._DeviceRng(cp, 11), xd, zd, nperm, nr, narrow=True)
    assert wide.dtype == cp.int64 and narrow.dtype == cp.uint8
    np.testing.assert_array_equal(cp.asnumpy(narrow).astype(np.int64), cp.asnumpy(wide))
    got = cp.asnumpy(narrow).astype(np.int64)
    for s in np.unique(z)[:200]:
        rows = np.flatnonzero(z == s)
        np.testing.assert_array_equal(np.sort(got[0, rows]), np.sort(x[rows]))
