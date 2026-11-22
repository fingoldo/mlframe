"""Column-block sizing reads the pool the thread allocates from: inside a limited slot pool the block shrinks to the slot, outside it follows the default pool."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_gpu_vram import current_memory_pool, pool_column_block


def test_current_pool_is_the_allocator_pool_inside_and_default_outside():
    """Inside ``using_allocator`` the thread's pool is the slot pool; outside it is the default pool."""
    slot = cp.cuda.MemoryPool()
    assert current_memory_pool(cp) is cp.get_default_memory_pool()
    with cp.cuda.using_allocator(slot.malloc):
        assert current_memory_pool(cp) is slot


def test_block_width_follows_the_slot_limit():
    """A 300 MB slot gives a narrower block than the default pool for the same rows."""
    kw = dict(bytes_per_column_row=64, budget_fraction=0.3, min_block=1, fallback=32, what="test", max_block=4096)
    outside = pool_column_block(cp, 100_000, **kw)
    slot = cp.cuda.MemoryPool()
    slot.set_limit(size=300 * 1024 * 1024)
    with cp.cuda.using_allocator(slot.malloc):
        inside = pool_column_block(cp, 100_000, **kw)
    assert inside == int(0.3 * 300 * 1024 * 1024) // (64 * 100_000)
    assert inside < outside
