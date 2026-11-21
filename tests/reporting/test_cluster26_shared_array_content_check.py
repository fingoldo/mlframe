"""Audit 2026-11-20 finding 26-F20: an auto-keyed share() of a buffer rewritten in place returns the new content.

The auto key is the buffer's address, so an in-place overwrite kept the old segment and a second share returned stale data. The
test shares, rewrites the buffer in place, shares again, and checks the second descriptor holds the new values while the first still
holds the old ones.
"""

import numpy as np

from mlframe.reporting._shared_arrays import SharedArrayPool


def test_rewritten_buffer_gets_its_own_segment_with_the_new_content():
    """The second share sees the new values under a new name; the first snapshot is unchanged."""
    pool = SharedArrayPool()
    try:
        buf = np.arange(6, dtype=np.float64)
        first = pool.share(buf)
        old_values = pool.owner_view(first).copy()
        buf[:] = buf + 100.0
        second = pool.share(buf)
        assert second.name != first.name
        assert np.array_equal(pool.owner_view(second), buf)
        assert np.array_equal(pool.owner_view(first), old_values)
    finally:
        pool.close()


def test_unchanged_buffer_still_reuses_its_segment():
    """Sharing an unchanged buffer twice returns the same segment, so the copy is not repeated."""
    pool = SharedArrayPool()
    try:
        buf = np.arange(4, dtype=np.float64)
        assert pool.share(buf).name == pool.share(buf).name
    finally:
        pool.close()
