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


def test_sort_room_reserved_against_a_concurrent_allocation():
    """Two threads of one limited pool: one sorts while the other keeps allocating; the sort never aborts the process, it sorts or raises OutOfMemoryError."""
    import threading

    _require_device()
    from mlframe.feature_selection.filters._device_sort_safe import device_argsort

    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=96 << 20)
    outcomes = []

    def hog():
        """Allocate 2 MB blocks from the shared pool until it is full, then release them."""
        with cp.cuda.using_allocator(pool.malloc):
            held = []
            try:
                for _ in range(40):
                    held.append(cp.empty(2 << 20, dtype=cp.uint8))
            except cp.cuda.memory.OutOfMemoryError:
                pass
            del held

    def sorter():
        """Argsort a 1M-element column five times on the shared pool, recording sorted or oom."""
        with cp.cuda.using_allocator(pool.malloc):
            x = cp.random.random(1_000_000).astype(cp.float32)
            for _ in range(5):
                try:
                    outcomes.append(bool((x[device_argsort(x)][1:] >= x[device_argsort(x)][:-1]).all()))
                except cp.cuda.memory.OutOfMemoryError:
                    outcomes.append("oom")

    threads = [threading.Thread(target=hog), threading.Thread(target=sorter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes and all(o is True or o == "oom" for o in outcomes)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_need_estimate_covers_axis0_matrix_sort(dtype):
    """The reserved room covers the measured pool high-water of an axis-0 argsort of a column matrix (float32 needed more than the old estimate gave)."""
    _require_device()
    from mlframe.feature_selection.filters._device_sort_safe import _sort_need_bytes

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        a = cp.random.random((20000, 16)).astype(dtype)
        base = pool.total_bytes()
        idx = cp.argsort(a, axis=0)
        cp.cuda.Device().synchronize()
        peak = pool.total_bytes() - base
        need = _sort_need_bytes(a)
        del a, idx
    pool.free_all_blocks()
    assert need >= peak
