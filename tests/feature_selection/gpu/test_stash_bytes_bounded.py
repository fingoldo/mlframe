"""Device codes parked for a lazy host fill are bounded by bytes (a share of the pool limit), not only by entry count; an evicted deferred entry is filled first."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_shared as S


def test_parked_codes_stay_within_the_byte_budget_and_evicted_entries_are_filled(monkeypatch):
    """Four 1 MB code blocks with a 2.5 MB budget: at most two stay parked, the evicted ones reach their host buffer unchanged."""
    monkeypatch.setattr(S, "_stash_byte_budget", lambda: int(2.5 * 2**20))
    S._DEFERRED_HOST_FILL.clear()
    S._RESIDENT_CODES_HANDOFF.clear()
    hosts, devs = [], []
    for i in range(4):
        dev = cp.full((1024, 1024), i + 1, dtype=cp.uint8)
        host = np.empty((1024, 1024), dtype=np.uint8)
        S._stash_resident_codes(host, dev)
        S._stash_deferred_host_fill(host, dev)
        hosts.append(host)
        devs.append(dev)
    with S._DEFERRED_HOST_FILL_LOCK:
        assert S._parked_device_bytes() <= int(2.5 * 2**20)
        assert len(S._DEFERRED_HOST_FILL) <= 2
    assert int(hosts[0][0, 0]) == 1 and int(hosts[1][5, 5]) == 2  # the evicted entries were filled before they were dropped
    S._DEFERRED_HOST_FILL.clear()
    S._RESIDENT_CODES_HANDOFF.clear()
