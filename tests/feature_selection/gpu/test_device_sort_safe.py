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


def test_full_device_raises_out_of_memory_even_with_pool_room(monkeypatch):
    """The device itself short (other processes or pools hold it) while the sort's own pool has no limit: the helper raises OutOfMemoryError
    (thrust aborts on a failing device allocation as on a pool limit). The device's free memory is faked: filling the real device raced the other
    workers of a parallel test run, which allocate and free on the same card."""
    _require_device()
    from mlframe.feature_selection.filters._device_sort_safe import device_argsort

    a = cp.asarray(np.random.default_rng(2).random((400, 5000)).astype(np.float64))  # 16 MB; its sort needs about 240 MB
    total = int(cp.cuda.runtime.memGetInfo()[1])
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (64 << 20, total))  # 64 MB free on the device
    pool = cp.cuda.MemoryPool()  # no limit: only the device can be short
    with cp.cuda.using_allocator(pool.malloc):
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            device_argsort(a, axis=1)
