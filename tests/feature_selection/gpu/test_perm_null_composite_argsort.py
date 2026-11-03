"""The permutation null's column argsort through one composite 1-D sort equals cupy's axis-0 argsort, ties included."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_cmi_perm_null_gpu import _column_argsort


@pytest.mark.parametrize("n,p,bits", [(50000, 25, 20), (100000, 7, 25), (300000, 1, 12)])
def test_equals_axis0_argsort(n, p, bits):
    """Stratum ranks in the high bits, random low bits, forced ties in a column."""
    rs = cp.random.RandomState(0)
    zr = cp.sort(cp.asarray(np.random.default_rng(0).integers(0, 1 << (31 - bits), n)).astype(cp.int32))
    keys = (zr << bits)[:, None] | rs.randint(0, 1 << bits, size=(n, p), dtype=cp.int32)
    keys[::7, 0] = keys[0, 0]
    assert cp.array_equal(_column_argsort(cp, keys), cp.argsort(keys, axis=0))
