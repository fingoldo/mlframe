"""Deterministic write-extent regression for the ``dedup_njit_edges`` CUDA kernel.

The kernel (``_fe_batched_mi._DEDUP_EDGES_SRC``) emits up to ``ne = nbins-1`` distinct interior edges into
``out[0:w, c]`` and then APPENDS the ``cmax`` row at write index ``w`` BEFORE the ``w -= 1`` decrement. For an
all-distinct continuous column ``w`` reaches ``ne`` after the loop, so the ``cmax`` append writes ``out[ne*K+c]``
-- index ``ne``, one row past an ``(ne, K)`` buffer. In a pooled fit that stray 8-byte write lands on the
adjacent ``ne_k`` allocation, corrupting a length to a garbage-huge int -> the length-aware MI kernel then
binary-searches with a multi-billion bound -> a multi-terabyte OOB read -> ``cudaErrorIllegalAddress`` ->
CUDA-context corruption. The caller therefore sizes ``Ec`` as ``(ne+1, K)``.

This test pins the kernel's WRITE EXTENT directly (no reliance on mempool layout, unlike a corruption-repro
which CUDA arena-rounding masks): run the kernel into a sentinel-filled ``(ne+2, K)`` buffer and assert it
writes exactly through row ``ne`` (the ``cmax`` row) and NEVER row ``ne+1``. A buffer of ``(ne, K)`` would thus
overflow; ``(ne+1, K)`` is the minimum safe size."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Need cuda."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_dedup_edges_writes_exactly_ne_plus_one_rows():
    """Dedup edges writes exactly ne plus one rows."""
    from mlframe.feature_selection.filters._fe_batched_mi import _get_dedup_edges_kernel

    nbins = 10
    ne = nbins - 1  # interior-edge count
    K = 8
    SENT = -123456.0

    # All-distinct, strictly increasing interior edges per column (1..ne), cmin below, cmax above the last
    # edge -> the loop writes all ne edges (w: 0->ne) and the cmax append fires at w == ne (the overflow row).
    edges = cp.ascontiguousarray(cp.tile(cp.arange(1, ne + 1, dtype=cp.float64)[:, None], (1, K)))  # (ne, K)
    cmin = cp.zeros(K, dtype=cp.float64)
    cmax = cp.full(K, float(ne + 1), dtype=cp.float64)

    # Sentinel-filled (ne+2, K): row ne is the legal cmax-append slot, row ne+1 must stay untouched.
    out = cp.full((ne + 2, K), SENT, dtype=cp.float64)
    ne_out = cp.empty(K, dtype=cp.int32)

    threads = 256
    _get_dedup_edges_kernel()(((K + threads - 1) // threads,), (threads,), (edges, cmin, cmax, np.int32(ne), np.int32(K), out, ne_out))
    cp.cuda.Stream.null.synchronize()

    out_h = cp.asnumpy(out)
    # Row ne (the cmax append) IS written for every column -> the kernel touches index ne -> an (ne, K)
    # buffer would overflow; (ne+1, K) is required.
    assert np.all(out_h[ne, :] == float(ne + 1)), "cmax row at index ne must be written (proves ne+1 needed)"
    # Row ne+1 is NEVER written -> the kernel never exceeds index ne, so (ne+1, K) is also SUFFICIENT.
    assert np.all(out_h[ne + 1, :] == SENT), "kernel must not write past row ne"
    # Reported length is ne (njit de[1:-1] parity), bounded by the buffer.
    assert np.all(cp.asnumpy(ne_out) == ne)
