"""Stage slots: heaviest stage first on the device that finishes it soonest, memory split by weight, refusal when a stage would starve, and a private pool per thread."""

import threading

import pytest

from mlframe.feature_selection.filters._fe_device_scheduler import DeviceSlot, plan_stage_slots, stage_device_slot
from tests._cuda_require import require_free_vram

GB = 1 << 30


def test_single_device_splits_usable_memory_by_weight():
    """Three stages on one 4 GB card with a 1 GB cushion: budgets follow the weights and sum to the usable memory."""
    slots = plan_stage_slots([30.0, 10.0, 20.0], facts=[(0, 4 * GB, 6)])
    assert [s.device_id for s in slots] == [0, 0, 0]
    assert sum(s.budget_bytes for s in slots) <= 3 * GB
    assert slots[0].budget_bytes > slots[2].budget_bytes > slots[1].budget_bytes


def test_heterogeneous_devices_put_the_heavy_stage_on_the_larger_card():
    """A 24 GB / 80 SM card and a 4 GB / 6 SM card: the heaviest stage goes to the big one, and no budget exceeds its device's usable memory."""
    facts = [(0, 4 * GB, 6), (1, 24 * GB, 80)]
    slots = plan_stage_slots([34.0, 19.0, 12.0], facts=facts)
    assert slots[0].device_id == 1
    for dev, free_b, _s in facts:
        assert sum(s.budget_bytes for s in slots if s.device_id == dev) <= free_b - GB


def test_a_starved_plan_is_refused():
    """With less usable memory than the minimum budget per stage the planner returns None, so the cascade stays sequential."""
    assert plan_stage_slots([1.0, 1.0, 1.0], facts=[(0, GB + (300 << 20), 6)]) is None
    assert plan_stage_slots([1.0], facts=[]) is None
    assert plan_stage_slots([], facts=[(0, 4 * GB, 6)]) == []


def test_slot_budget_is_enforced_per_thread():
    """A thread inside a 300 MB slot cannot allocate 400 MB, while another thread with its own slot allocates 200 MB at the same time."""
    cp = pytest.importorskip("cupy")
    require_free_vram(1 << 30)
    results = {}
    barrier = threading.Barrier(2)

    def small():
        """Allocate within the budget and keep it alive while the other thread tries to exceed its own."""
        with stage_device_slot(DeviceSlot(0, 300 << 20)):
            a = cp.empty(200 << 20, dtype=cp.uint8)
            barrier.wait(timeout=60)
            barrier.wait(timeout=60)
            results["small"] = int(a.nbytes)

    def big():
        """Try to exceed the budget of this thread's slot."""
        with stage_device_slot(DeviceSlot(0, 300 << 20)):
            barrier.wait(timeout=60)
            try:
                cp.empty(400 << 20, dtype=cp.uint8)
                results["big"] = "allocated"
            except cp.cuda.memory.OutOfMemoryError:
                results["big"] = "refused"
            barrier.wait(timeout=60)

    threads = [threading.Thread(target=small), threading.Thread(target=big)]
    [t.start() for t in threads]
    [t.join(timeout=120) for t in threads]
    assert results == {"small": 200 << 20, "big": "refused"}


def test_slot_releases_its_blocks_on_exit():
    """After the context the pool's cached blocks are returned to the device."""
    cp = pytest.importorskip("cupy")
    before, _t = cp.cuda.runtime.memGetInfo()
    with stage_device_slot(DeviceSlot(0, 400 << 20)) as pool:
        x = cp.empty(100 << 20, dtype=cp.uint8)
        del x
        assert pool.total_bytes() >= (100 << 20)
    after, _t = cp.cuda.runtime.memGetInfo()
    assert after >= before - (50 << 20)
