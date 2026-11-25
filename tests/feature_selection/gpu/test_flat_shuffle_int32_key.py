"""The within-stratum flat shuffle keeps each stratum's values and moves them: the int32-key branch and the int64-key branch (MRMR-MATERIALISE, 2026-11-19)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

import mlframe.feature_selection.filters._fe_cmi_perm_null_gpu as G


def _data(n: int, nrank: int, kx: int, seed: int):
    """Rows sorted by stratum: ``x`` codes below ``kx`` and the dense stratum rank column ``(n, 1)``."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.integers(0, nrank, n))
    _, z = np.unique(z, return_inverse=True)
    x = rng.integers(0, kx, n).astype(np.int64)
    return x, z.astype(np.float64)[:, None]


@pytest.mark.parametrize("n, nperm, nrank, kx, branch", [(6000, 40, 15, 2, "int32"), (4000, 8, 5, 3, "int32"), (6000, 25, 128, 10, "int64")])
def test_each_stratum_keeps_its_values_and_the_order_moves(n: int, nperm: int, nrank: int, kx: int, branch: str):
    """Per permutation and stratum the multiset of values equals the input's; across permutations the layout differs; both key widths are exercised."""
    x, z = _data(n, nrank, kx, 3)
    rs = G._DeviceRng(cp, 7)
    out = G._flat_within_stratum_shuffle(cp, rs, cp.asarray(x), cp.asarray(z), nperm, kx, int(z.max()) + 1)
    assert out is not None
    assert out.dtype == (cp.int32 if branch == "int32" else cp.int64), out.dtype
    got = cp.asnumpy(out)
    assert got.shape == (nperm, n)
    zr = z[:, 0].astype(np.int64)
    for s in np.unique(zr):
        rows = np.flatnonzero(zr == s)
        ref = np.sort(x[rows])
        for p in range(nperm):
            np.testing.assert_array_equal(np.sort(got[p, rows]), ref, err_msg=f"stratum {s} perm {p}")
    assert not np.array_equal(got[0], got[1]), "two permutations are identical"
    assert not np.array_equal(got[0], x), "a permutation left the layout unchanged"
