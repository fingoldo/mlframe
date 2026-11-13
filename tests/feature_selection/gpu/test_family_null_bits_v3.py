"""The permutation-blocked family null kernel (v3) gives exactly the v2 results.

v2 re-reads every column's bin bit sets once per permutation (40 permutations x 100 MB = 4 GB for a 9 ms call). v3 gives a block ``PB`` permutations and loads each bin word once for all of them, with the class bit sets
staged in shared memory in tiles of 128 words. The per-pair sums are accumulated in v2's order, so the maxima and the MI matrix must equal v2's bit for bit, for row counts that leave a short last word and a short
last tile, odd permutation counts (a short last block), columns with different level counts, and two or three label classes.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _family_maxt_cuda_bits as fb
from mlframe.feature_selection.filters._family_maxt_gate import _label_const, bitset_groups
from mlframe.feature_selection.filters._family_maxt_null_v2_njit import label_bitsets_batch


def _setup(n, k, ny, b, seed, max_levels=8):
    """Device bit sets of ``k`` columns with 2..max_levels levels each, ``b`` permuted label vectors with ``ny`` classes."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(2, max_levels + 1, k)
    cols = np.stack([rng.integers(0, int(lv), n) for lv in levels]).astype(np.int64)
    nx = (cols.max(axis=1) + 1).astype(np.int64)
    y = rng.integers(0, ny, n).astype(np.int64)
    perms = np.stack([y[rng.permutation(n)] for _ in range(b)]).astype(np.int64)
    prep = bitset_groups(cols, nx)
    dev = fb.DeviceBitsets(cp, prep, nx, n)
    cls = label_bitsets_batch(perms, ny, (n + 63) // 64)
    return dev, cls, _label_const(prep[0], perms[0], ny), perms


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("n,k,ny,b", [(6437, 33, 2, 7), (129, 5, 3, 5), (20_000, 70, 2, 4), (300_000 // 20, 129, 2, 9), (1_000, 3, 3, 3)])
def test_v3_equals_v2_exactly(n, k, ny, b, seed, monkeypatch):
    """Per-permutation maxima and the full MI matrix equal v2's, bit for bit."""
    dev, cls, const, _perms = _setup(n, k, ny, b, seed)
    assert fb._v3_plan(ny, dev.cap) is not None or ny > 2
    got_max = fb.null_max_bits_device(dev, cls, const)
    const_b = np.full(b, const, dtype=np.float64)
    got_mat = fb.mi_matrix_bits_device(dev, cls, const_b)
    monkeypatch.setenv("MLFRAME_FAMILY_BITS_V3", "0")
    ref_max = fb.null_max_bits_device(dev, cls, const)
    ref_mat = fb.mi_matrix_bits_device(dev, cls, const_b)
    np.testing.assert_array_equal(got_max, ref_max)
    np.testing.assert_array_equal(got_mat, ref_mat)


def test_the_plan_gives_v2_when_the_counters_do_not_fit_registers():
    """Eight bins with four classes need 24 counters for one permutation, so the blocked kernel declines; two classes at eight bins take four permutations per block."""
    assert fb._v3_plan(4, 8) is None
    assert fb._v3_plan(2, 8) == 4
    assert fb._v3_plan(3, 8) == 2
    assert fb._v3_plan(1, 8) is None


def test_the_switch_forces_v2(monkeypatch):
    """``MLFRAME_FAMILY_BITS_V3=0`` selects v2 whatever the shape."""
    monkeypatch.setenv("MLFRAME_FAMILY_BITS_V3", "0")
    assert fb._v3_plan(2, 8) is None
