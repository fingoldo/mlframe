"""The heartbeat must stop with the suite and restart on the next one (a notebook process outlives the suite)."""

from __future__ import annotations

import time

from mlframe.training import crash_diagnostics as cd


def test_stop_then_restart():
    hb = cd.start_heartbeat(3600)
    assert hb is not None and hb.alive
    cd.stop_heartbeat()
    time.sleep(0.2)
    assert not hb.alive
    hb2 = cd.start_heartbeat(3600)
    assert hb2 is not None and hb2.alive and hb2 is not hb
    cd.stop_heartbeat()
