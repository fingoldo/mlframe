"""The stage slots' memory pools live for the whole process: device arrays allocated inside them are held by process-global caches, so a pool must not be dropped when a stage ends."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_device_scheduler as sched


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver / no device: the test is skipped
        return False


@pytest.mark.skipif(not _device_ok(), reason="no CUDA device")
def test_slot_pool_survives_its_stage_and_keeps_cached_arrays_alive():
    """An array allocated in a slot pool and held after the stage ends still reads back its values once the stage's pool is asked for again."""
    slot = sched.DeviceSlot(device_id=0, budget_bytes=64 << 20)
    with sched.stage_device_slot(slot, 91) as pool:
        held = cp.arange(1000, dtype=cp.int64)
    assert sched._SLOT_POOLS[(0, 91)] is pool
    with sched.stage_device_slot(slot, 91) as again:
        other = cp.full(1000, -7, dtype=cp.int64)
        assert again is pool
        assert int(held.sum()) == 499500 and int(other.sum()) == -7000
