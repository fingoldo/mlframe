"""The shared-memory binary-search qbin coder gives the codes of the broadcast loop (count of distinct interior edges <= x, plus the two-distinct special case) for float32 and double, byte and int64 codes, narrow and wide column counts (several slabs), constant, binary and tied columns."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _fe_batched_qbin as q


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def _reference(cp, X, nbins, single):
    """The broadcast loop the fused coder replaced, on the same edges."""
    edges = q._edges_of_block(cp, X, nbins, single, True)
    first = cp.ones(edges.shape, dtype=cp.bool_)
    first[1:] = edges[1:] != edges[:-1]
    interior = first & (edges > edges[0][None]) & (edges < edges[-1][None])
    nd = first.sum(axis=0)
    codes = cp.zeros(X.shape, dtype=cp.int64)
    for j in range(1, edges.shape[0]):
        codes += (interior[j][None] & (X >= edges[j][None])).astype(cp.int64)
    codes += ((nd == 2)[None] & (X >= edges[-1][None])).astype(cp.int64)
    return codes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,K,nbins", [(5000, 7, 10), (20_000, 100, 20), (3001, 700, 20), (1000, 33, 64)])
@pytest.mark.parametrize("byte_codes", [False, True])
def test_codes_equal_the_broadcast_loop(dtype, n, K, nbins, byte_codes):
    """Gaussian columns plus a constant, a binary and a heavily tied column."""
    cp = _cp()
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, K)).astype(dtype)
    X[:, 0] = 3.0
    X[:, 1] = rng.integers(0, 2, n)
    X[:, 2] = np.round(X[:, 2] * 2) / 2
    Xd = cp.asarray(X)
    got = q.batched_quantile_bin_gpu(Xd, nbins, byte_codes=byte_codes)
    assert got.dtype == (cp.uint8 if byte_codes and nbins <= 255 else cp.int64)
    want = _reference(cp, Xd, nbins, dtype == np.float32)
    np.testing.assert_array_equal(cp.asnumpy(got).astype(np.int64), cp.asnumpy(want))
