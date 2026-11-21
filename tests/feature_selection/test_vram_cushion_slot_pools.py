"""The VRAM cushion check releases the idle cached blocks of a stage slot pool, not only those of the default pool."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_gpu_vram as V


def test_own_pools_include_the_pool_of_the_calling_thread():
    """Inside a stage slot the allocator pool is one of the pools the check looks at."""
    slot = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(slot.malloc):
        pools = V._own_pools(cp)
    assert any(p is slot for p in pools) and any(p is cp.get_default_memory_pool() for p in pools)
    assert len({id(p) for p in pools}) == len(pools)


def test_idle_blocks_of_a_slot_pool_are_counted_and_released():
    """A slot pool that holds a freed 64 MB block reports it and gives it back to the device."""
    slot = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(slot.malloc):
        x = cp.empty(64 * 1024 * 1024, dtype=cp.uint8)
        del x
        assert slot.free_bytes() >= 64 * 1024 * 1024
        assert V._own_pool_free_bytes() >= 64 * 1024 * 1024
        V._release_own_pool()
        assert slot.free_bytes() == 0
