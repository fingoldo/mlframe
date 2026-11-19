"""The fused MI-from-counts reduction gives the MI of the former cupy chain to rounding, and the same ranking (MRMR-LAUNCHCOUNT, 2026-11-18)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_batched import _mi_from_counts_cupy


def _chain(counts_2d, n, nbins_x, nbins_y, outer_safe, dtype):
    """The former chain, kept as the reference: astype, divide, where, log, multiply, sum."""
    jf = (counts_2d.astype(dtype) / n).reshape(-1, nbins_x, nbins_y)
    ratio = cp.where(jf > 0, jf / outer_safe, 1.0)
    return (jf * cp.log(ratio)).sum(axis=(1, 2))


@pytest.mark.parametrize("b, nbins_x, nbins_y", [(1, 10, 2), (64, 10, 2), (35, 10, 3)])
def test_fused_mi_matches_the_former_chain_and_ranks_the_same(b: int, nbins_x: int, nbins_y: int):
    """Random counts with empty cells and a zero outer marginal: values agree to 1e-12 and the argmax (the selection) is identical."""
    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype as _dt
    rng = np.random.default_rng(b + nbins_x)
    n = 5000
    counts = rng.integers(0, 40, size=(b, nbins_x * nbins_y)).astype(np.int32)
    counts[:, 0] = 0
    outer_np = rng.integers(0, 50, size=(1, nbins_x, nbins_y)).astype(np.float64)
    outer_np[outer_np == 0] = 0
    dtype = np.dtype(_dt())
    outer_safe = cp.where(cp.asarray(outer_np.astype(dtype)) > 0, cp.asarray(outer_np.astype(dtype)), 1.0)
    dev = cp.asarray(counts)
    fused = cp.asnumpy(_mi_from_counts_cupy(cp, dev, n, nbins_x, nbins_y, outer_safe))
    ref = cp.asnumpy(_chain(dev, n, nbins_x, nbins_y, outer_safe, dtype))
    np.testing.assert_allclose(fused, ref, rtol=1e-6, atol=1e-9)
    assert int(np.argmax(fused)) == int(np.argmax(ref))
