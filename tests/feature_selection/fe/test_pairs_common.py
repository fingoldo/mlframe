"""Unit coverage for ``_feature_engineering_pairs/_pairs_common.py``'s shared module-scope state.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this tiny leaf module (the shared logger +
``times_spent`` accumulator lock for the FE pair-search submodules) had zero test references anywhere
in the suite. Pins the two properties that matter: the lock is a genuine mutual-exclusion lock shared
by identity across every importer (not a fresh lock per import), and the logger resolves to the
documented dotted name.
"""

from __future__ import annotations

import logging
import threading

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_common
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_common import (
    _TIMES_SPENT_LOCK,
    _module_logger,
)


def test_module_logger_has_documented_name():
    """The shared logger resolves to the documented dotted name, not ``__name__`` of some submodule."""
    assert isinstance(_module_logger, logging.Logger)
    assert _module_logger.name == "mlframe.feature_selection.filters._feature_engineering_pairs"


def test_times_spent_lock_is_a_real_mutex():
    """``_TIMES_SPENT_LOCK`` actually serializes: a second acquire from another thread blocks until
    the first releases (this is the whole reason it exists -- see the module's own Wave 27 note about
    non-atomic float += across threads)."""
    acquired_second = threading.Event()
    released_first = threading.Event()

    def _holder():
        """Hold the lock briefly, signal release, then let go."""
        with _TIMES_SPENT_LOCK:
            released_first.wait(timeout=2.0)

    t = threading.Thread(target=_holder)
    t.start()
    try:
        # Give the holder thread a moment to actually acquire the lock first.
        import time

        time.sleep(0.05)
        assert _TIMES_SPENT_LOCK.acquire(blocking=False) is False, "lock should be held by the other thread"
    finally:
        released_first.set()
        t.join(timeout=2.0)
    # Now that the holder released it, this thread can acquire it.
    assert _TIMES_SPENT_LOCK.acquire(blocking=False) is True
    _TIMES_SPENT_LOCK.release()
    acquired_second.set()


def test_lock_and_logger_are_shared_by_identity_across_importers():
    """Every submodule that imports ``_TIMES_SPENT_LOCK`` / ``_module_logger`` gets the SAME object
    (not a fresh copy per import), which is the entire point of hoisting them to this shared leaf."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score

    assert _pairs_score._TIMES_SPENT_LOCK is _TIMES_SPENT_LOCK
    assert _pairs_score._TIMES_SPENT_LOCK is _pairs_common._TIMES_SPENT_LOCK
