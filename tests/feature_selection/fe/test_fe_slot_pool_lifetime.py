"""An array allocated inside a stage slot stays valid after the slot ends (process-global caches keep such arrays)."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_device_scheduler import DeviceSlot, stage_device_slot
from tests._cuda_require import require_cuda_device


def test_array_from_a_slot_survives_the_slot():
    """The slot pool is kept alive, so its in-use chunk is not handed out again while the cached array lives."""
    require_cuda_device()
    slot = DeviceSlot(device_id=0, budget_bytes=300 << 20)
    with stage_device_slot(slot, key=7):
        cached = cp.arange(1_000_000, dtype=cp.float64) * 2
        tmp = cp.ones(10_000_000)
        del tmp
    reuse = cp.ones(10_000_000)
    got = float(cached[-1])
    del reuse
    assert got == 999_999 * 2.0
