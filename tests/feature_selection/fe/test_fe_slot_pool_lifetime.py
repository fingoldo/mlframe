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


def test_reentered_slot_budget_is_on_top_of_held_arrays():
    """A slot entered again while its pool still holds an earlier fit's cached array gets its full budget as new room."""
    require_cuda_device()
    slot = DeviceSlot(device_id=0, budget_bytes=64 << 20)
    with stage_device_slot(slot, key=8):
        held = cp.ones(6 << 20)  # 48 MiB that outlive the slot, as cached device columns do
    with stage_device_slot(slot, key=8) as pool:
        fresh = cp.ones(6 << 20)  # needs 48 MiB more: over a 64 MiB limit that counted the held array, within one on top of it
        assert pool.get_limit() >= pool.used_bytes()
    del fresh
    assert float(held[0]) == 1.0


def test_fit_end_release_returns_cached_slot_blocks():
    """Blocks freed after their slot ended stay cached in the persistent slot pool until the fit-end release returns them to the device."""
    require_cuda_device()
    from mlframe.feature_selection.filters._fit_scoped_release import release_fit_scoped_caches

    slot = DeviceSlot(device_id=0, budget_bytes=128 << 20)
    with stage_device_slot(slot, key=9) as pool:
        outlives = cp.ones(4 << 20)  # a cached device column: still alive when the slot frees its blocks
    del outlives
    assert pool.total_bytes() > 0
    release_fit_scoped_caches()
    assert pool.total_bytes() == 0


def test_fit_end_release_drops_resident_codes_handoff():
    """A resident-codes handoff entry left after a fit is dropped by the fit-end release, so its device codes do not outlive the fit."""
    require_cuda_device()
    import numpy as np

    from mlframe.feature_selection.filters import _gpu_resident_fe  # noqa: F401  (the releaser lives here; a fit always imports it)
    from mlframe.feature_selection.filters import _gpu_resident_shared as shared
    from mlframe.feature_selection.filters._fit_scoped_release import release_fit_scoped_caches

    host = np.zeros((4, 3), dtype=np.int8)
    shared._RESIDENT_CODES_HANDOFF[id(host)] = (cp.zeros((4, 3), dtype=cp.int8), host.shape, host.dtype, None)
    release_fit_scoped_caches()
    assert id(host) not in shared._RESIDENT_CODES_HANDOFF
