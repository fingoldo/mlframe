"""``_transpose_to_cm`` refuses at once when the pool limit leaves no room for the copy, and still transposes when it does."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_select_kernels import _transpose_to_cm


def _pool(limit):
    """A memory pool with a limit in bytes."""
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=limit)
    return pool


def test_no_room_raises_out_of_memory_without_touching_the_pool():
    """The copy needs 4 MB, the pool holds the 4 MB input under a 6 MB limit: OutOfMemoryError up front, the pool's cache is left alone."""
    pool = _pool(6 * 1024 * 1024)
    with cp.cuda.using_allocator(pool.malloc):
        x = cp.arange(1024 * 1024, dtype=cp.float32).reshape(1024, 1024)
        keep = cp.empty(1024 * 1024, dtype=cp.uint8)  # cached block that a failed malloc would have flushed
        del keep
        total_before = pool.total_bytes()
        with pytest.raises(cp.cuda.memory.OutOfMemoryError, match="Out of memory"):
            _transpose_to_cm(x)
        assert pool.total_bytes() == total_before


def test_with_room_it_transposes():
    """A pool with room gives the exact transpose."""
    pool = _pool(64 * 1024 * 1024)
    with cp.cuda.using_allocator(pool.malloc):
        a = np.random.default_rng(0).standard_normal((2000, 37)).astype(np.float32)
        out = _transpose_to_cm(cp.asarray(a))
        assert np.array_equal(cp.asnumpy(out), a.T)
