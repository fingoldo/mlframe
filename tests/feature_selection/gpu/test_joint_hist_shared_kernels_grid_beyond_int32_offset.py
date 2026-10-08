"""The shared-memory joint-histogram kernels must index rows in 64 bits: a grid whose first-row offset passes 2**31 must not wrap negative."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_BLOCK = 256
# blockIdx.x * 256 passes 2**31 at blockIdx.x = 2**23; the extra blocks past it are empty (their rows are >= n).
_GRID_X = (1 << 23) + 2048


@pytest.fixture(scope="module")
def kernels():
    """The lazily compiled joint-histogram kernels, skipped when no usable CUDA device exists."""
    from mlframe.feature_selection.filters import gpu as g

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:  # no device or driver
        pytest.skip(f"CUDA unavailable: {exc}")
    g._ensure_kernels_inited()
    return g


def test_batched_shared_kernel_ignores_blocks_whose_row_offset_exceeds_int32(kernels):
    """Blocks past the data (offset above 2**31) add nothing, and the counts equal the host histogram."""
    n, nbx, nby = 1000, 4, 3
    rng = np.random.default_rng(0)
    cx = rng.integers(0, nbx, n).astype(np.int32)
    cy = rng.integers(0, nby, n).astype(np.int32)
    out = cp.zeros((1, nbx * nby), dtype=cp.int32)
    kernels.compute_joint_hist_batched_shared_cuda(
        (_GRID_X, 1), (_BLOCK,), (cp.asarray(cx), cp.asarray(cy).reshape(1, n), out, np.int32(n), np.int32(nbx), np.int32(nby)), shared_mem=nbx * nby * 4
    )
    cp.cuda.runtime.deviceSynchronize()
    expected = np.zeros(nbx * nby, dtype=np.int32)
    np.add.at(expected, cx * nby + cy, 1)
    assert np.array_equal(cp.asnumpy(out)[0], expected)


def test_multi_pair_shared_kernel_ignores_blocks_whose_row_offset_exceeds_int32(kernels):
    """Same contract for the multi-pair shared kernel: empty far blocks, counts equal the host histogram."""
    n, nb, nby = 1000, 3, 2
    rng = np.random.default_rng(1)
    cols = rng.integers(0, nb, (2, n)).astype(np.int32)
    cy = rng.integers(0, nby, n).astype(np.int32)
    size = nb * nb * nby
    offsets = np.array([0, size], dtype=np.int32)
    out = cp.zeros(size, dtype=cp.int32)
    kernels.compute_joint_hist_multi_pair_shared_cuda(
        (_GRID_X, 1),
        (_BLOCK,),
        (
            cp.asarray(cols),
            cp.asarray(cy),
            cp.asarray(np.array([0], dtype=np.int32)),
            cp.asarray(np.array([1], dtype=np.int32)),
            cp.asarray(np.array([nb], dtype=np.int32)),
            cp.asarray(offsets),
            out,
            np.int32(n),
            np.int32(1),
            np.int32(nby),
            np.int32(size),
        ),
        shared_mem=size * 4,
    )
    cp.cuda.runtime.deviceSynchronize()
    merged = cols[0] + cols[1] * nb
    expected = np.zeros(size, dtype=np.int32)
    np.add.at(expected, merged * nby + cy, 1)
    assert np.array_equal(cp.asnumpy(out), expected)
