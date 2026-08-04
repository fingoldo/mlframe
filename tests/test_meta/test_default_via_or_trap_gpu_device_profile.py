"""default_via_or wave 3: fixes 3 P1 findings in ``feature_selection/filters/_fe_gpu_batch/_devices.py``
where ``_profile_device``'s own ``_p(key, default)`` helper already resolves a missing/None device-
properties key to ``default``, but the caller wrapped each ``_p(...)`` call in an additional
``or default`` -- silently rewriting a LEGITIMATELY-returned ``0`` (e.g. compute capability minor
version 0, as in sm_80/sm_90) to the same default, which for ``multiProcessorCount``/``clockRate``/
``major`` would corrupt the device profile with no signal that anything went wrong.
"""

from __future__ import annotations

import sys
import types
from unittest import mock


def test_profile_device_does_not_clobber_legitimate_zero_properties():
    """A device-properties value of exactly 0 for multiProcessorCount/clockRate/major must survive
    into the DeviceProfile unchanged, not be silently rewritten to the ``_p`` default."""
    fake_cupy = types.ModuleType("cupy")

    class _FakeDeviceCtx:
        """Minimal context-manager stand-in for ``cupy.cuda.Device``."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_runtime = mock.Mock()
    fake_runtime.memGetInfo.return_value = (1024, 2048)
    fake_runtime.getDeviceProperties.return_value = {
        "multiProcessorCount": 0,
        "clockRate": 0,
        "sharedMemPerBlock": None,  # missing -> must fall back to the _p default
        "major": 0,
        "minor": 0,
    }
    fake_cuda = types.SimpleNamespace(Device=lambda device: _FakeDeviceCtx(), runtime=fake_runtime)
    fake_cupy.cuda = fake_cuda

    with mock.patch.dict(sys.modules, {"cupy": fake_cupy}):
        from mlframe.feature_selection.filters._fe_gpu_batch._devices import _profile_device

        profile = _profile_device(0)

    assert profile.sm_count == 0, "multiProcessorCount=0 must survive, not be rewritten to the _p default of 1"
    assert profile.clock_khz == 0, "clockRate=0 must survive, not be rewritten to the _p default of 1"
    assert profile.cc_major == 0, "major=0 must survive, not be rewritten to the _p default of 6"
    assert profile.shared_per_block == 48 * 1024, "a genuinely-missing (None) key must still fall back to the _p default"
