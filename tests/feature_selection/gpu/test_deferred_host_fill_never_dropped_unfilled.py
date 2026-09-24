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
