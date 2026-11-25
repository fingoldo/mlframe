"""Cluster 26 regression: the shared error-prep build runs outside its lock (audit 2026-11-20, finding 26-F22).

A slow build for one target used to hold the prep lock, so an unrelated target waited for it. The lock now guards the cache only. While one
thread builds, a second thread with different arrays must finish without waiting.
"""

import threading

import numpy as np

from mlframe.reporting import _diagnostics_prep as prep


def test_unrelated_prep_is_not_blocked_by_a_slow_build():
    """A second target's prep completes while the first target's build is still running (26-F22)."""
    a_df, a_yt, a_yp = np.arange(20.0), np.arange(20.0), np.arange(20.0)
    b_df, b_yt, b_yp = np.arange(30.0), np.arange(30.0), np.arange(30.0)
    slow_started = threading.Event()
    release_slow = threading.Event()

    def slow_build():
        """Hold the build open until the test releases it."""
        slow_started.set()
        release_slow.wait(10)
        return (1,)

    slow = threading.Thread(target=lambda: prep.shared_error_prep(a_df, a_yt, a_yp, "regression", 0, slow_build))
    slow.start()
    try:
        assert slow_started.wait(5)
        fast = threading.Thread(target=lambda: prep.shared_error_prep(b_df, b_yt, b_yp, "regression", 0, lambda: (2,)))
        fast.start()
        fast.join(timeout=3)
        assert not fast.is_alive(), "the second prep waited for the first target's build"
    finally:
        release_slow.set()
        slow.join(timeout=10)
