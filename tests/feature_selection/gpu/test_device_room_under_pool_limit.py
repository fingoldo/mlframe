"""Block sizing reads the room of the pool behind the thread's allocator, not the whole card."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._device_sort_safe import device_room


def test_device_room_is_capped_by_the_slot_pool_limit():
    """Under a pool limited to 64 MB the room is at most the limit less what the pool already holds."""
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=64 * 1024 * 1024)
    previous = cp.cuda.get_allocator()
    cp.cuda.set_allocator(pool.malloc)
    try:
        held = cp.empty(8 * 1024 * 1024, dtype=cp.uint8)
        room = device_room(cp)
        assert 0 < room <= 64 * 1024 * 1024 - 8 * 1024 * 1024
        del held
    finally:
        cp.cuda.set_allocator(previous)
        pool.free_all_blocks()
