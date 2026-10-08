"""A deferred host-codes buffer is never left unfilled, however many are in flight.

The FE pair scoring bins codes on several threads, each registering an unfilled host buffer for a lazy D2H and then
waiting on the numba kernel lock. The registry is capped, and once more buffers were in flight than the cap, eviction
dropped the oldest record; ``ensure_host_codes_filled`` then did nothing and the CPU kernel read an ``np.empty``
buffer. Its garbage codes indexed out of bounds: an access violation on Windows, or MI computed on garbage.
"""

from __future__ import annotations

import threading

import numpy as np

from mlframe.feature_selection.filters import _gpu_resident_fe as gfe


class _FakeDevice:
    """Stands in for a cupy array: ``get(out=...)`` copies known codes into the host buffer."""

    def __init__(self, codes: np.ndarray) -> None:
        """Keep the codes a D2H would deliver."""
        self.codes = codes

    def get(self, out: np.ndarray) -> None:
        """Copy the codes into ``out``, as a device-to-host transfer would."""
        out[...] = self.codes


def _stash(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Register one unfilled host buffer whose device codes are all ``k``."""
    codes = np.full((50, 3), k % 10, dtype=np.int8)
    host = np.full((50, 3), -1, dtype=np.int8)  # what an unfilled np.empty buffer may hold
    gfe._stash_deferred_host_fill(host, _FakeDevice(codes))
    return host, codes


def test_buffers_past_the_cap_are_still_filled() -> None:
    """Register twice the cap before any consumer reads: every buffer must hold its codes once its consumer asks."""
    n = 2 * gfe._DEFERRED_HOST_FILL_MAX + 3
    pairs = [_stash(k) for k in range(n)]
    try:
        for host, codes in pairs:
            gfe.ensure_host_codes_filled(host)
            np.testing.assert_array_equal(host, codes)
    finally:
        for host, _ in pairs:
            gfe.clear_resident_codes_handoff(host)


def test_concurrent_stash_and_fill_leaves_no_buffer_unfilled() -> None:
    """Many threads stash and then fill at once, the way the pair scoring's pool does."""
    n = 4 * gfe._DEFERRED_HOST_FILL_MAX
    results: list = [None] * n
    barrier = threading.Barrier(n)

    def work(k: int) -> None:
        """Stash, wait until every thread has stashed, then fill and record the outcome."""
        host, codes = _stash(k)
        barrier.wait()
        gfe.ensure_host_codes_filled(host)
        results[k] = bool(np.array_equal(host, codes))
        gfe.clear_resident_codes_handoff(host)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results), f"{results.count(False)} of {n} buffers were read unfilled"


class _CountingDevice(_FakeDevice):
    """A fake device that counts how many device-to-host copies were requested."""

    def __init__(self, codes: np.ndarray) -> None:
        """Keep the codes and start the transfer count at zero."""
        super().__init__(codes)
        self.transfers = 0

    def get(self, out: np.ndarray) -> None:
        """Count the transfer, then copy as a real one would."""
        self.transfers += 1
        super().get(out)


def test_buffers_whose_owner_is_gone_are_dropped_without_a_transfer() -> None:
    """A record whose host buffer was garbage-collected has no reader: it is pruned, and eviction never copies its codes."""
    import gc

    devices = []
    for k in range(3 * gfe._DEFERRED_HOST_FILL_MAX):
        dev = _CountingDevice(np.full((50, 3), k % 10, dtype=np.int8))
        devices.append(dev)
        host = np.full((50, 3), -1, dtype=np.int8)
        gfe._stash_deferred_host_fill(host, dev)
        del host  # the dispatch ended without a host read
        gc.collect()
    assert sum(d.transfers for d in devices) == 0
    assert len(gfe._DEFERRED_HOST_FILL) <= 1


def test_a_live_owner_is_still_filled_when_dead_records_are_pruned() -> None:
    """Pruning dead records must not touch a buffer that is still referenced: that owner still gets its codes."""
    import gc

    live, live_codes = _stash(7)
    for k in range(2 * gfe._DEFERRED_HOST_FILL_MAX):
        _stash(k)  # their hosts are dropped immediately
    gc.collect()
    try:
        gfe.ensure_host_codes_filled(live)
        np.testing.assert_array_equal(live, live_codes)
    finally:
        gfe.clear_resident_codes_handoff(live)
