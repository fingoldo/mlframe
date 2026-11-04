"""A device sort under a pool with a hard limit raises cupy's OutOfMemoryError in Python instead of letting thrust abort the process."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _require_device():
    """Skip without a usable device."""
    try:
        cp.zeros(1)
    except Exception:
        pytest.skip("no usable CUDA device")


def test_short_pool_raises_out_of_memory_not_abort():
    """A pool limited below the sort's temporaries raises OutOfMemoryError from the helper (a raw cupy.argsort there aborts the process)."""
    _require_device()
    from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        a = cp.asarray(np.random.default_rng(0).random((200, 5000)).astype(np.float32))  # 4 MB
        pool.set_limit(size=int(pool.used_bytes()) + (1 << 20))  # 1 MB of room: far less than a sort needs
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            device_argsort(a, axis=1)
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            device_sort(a, axis=1)
    del a
    pool.free_all_blocks()


def test_enough_room_sorts_like_cupy():
    """With room the helpers return what cupy.argsort / cupy.sort return."""
    _require_device()
    from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort

    a = cp.asarray(np.random.default_rng(1).random((50, 300)).astype(np.float32))
    assert bool(cp.all(device_argsort(a, axis=1) == cp.argsort(a, axis=1)))
    assert bool(cp.all(device_sort(a, axis=0) == cp.sort(a, axis=0)))
