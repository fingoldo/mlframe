"""Unit coverage for ``_fe_deadline.py``'s thread-local enrichment-FE wall-clock deadline.

X_TEST_COVERAGE_QUALITY-5 fix (mrmr_audit_2026-07-22): this module had zero test references anywhere
in the suite despite gating every optional enrichment FE generator's internal per-column/per-pair
budget. Directly unit-testable: it is a plain thread-local, no fixture/fit needed.
"""

from __future__ import annotations

from timeit import default_timer as timer

from mlframe.feature_selection.filters._fe_deadline import (
    clear_fe_deadline,
    fe_budget_active,
    fe_deadline_passed,
    set_fe_deadline,
)


def teardown_function():
    """Never leak a deadline into another test on the same thread."""
    clear_fe_deadline()


def test_no_deadline_set_never_passes():
    """With no deadline set (the common no-budget path), ``fe_deadline_passed`` is always False and
    ``fe_budget_active`` is False."""
    clear_fe_deadline()
    assert fe_deadline_passed() is False
    assert fe_budget_active() is False


def test_future_deadline_not_yet_passed():
    """A deadline comfortably in the future is not yet passed, but the budget IS active."""
    set_fe_deadline(timer() + 60.0)
    assert fe_deadline_passed() is False
    assert fe_budget_active() is True


def test_past_deadline_has_passed():
    """A deadline already in the past is reported as passed."""
    set_fe_deadline(timer() - 1.0)
    assert fe_deadline_passed() is True
    assert fe_budget_active() is True


def test_set_deadline_none_disables_it():
    """Passing ``None`` to ``set_fe_deadline`` disables the deadline, same as ``clear_fe_deadline``."""
    set_fe_deadline(timer() - 1.0)
    assert fe_deadline_passed() is True
    set_fe_deadline(None)
    assert fe_deadline_passed() is False
    assert fe_budget_active() is False


def test_clear_fe_deadline_resets_state():
    """``clear_fe_deadline`` explicitly resets an active deadline to unset."""
    set_fe_deadline(timer() + 60.0)
    assert fe_budget_active() is True
    clear_fe_deadline()
    assert fe_budget_active() is False
    assert fe_deadline_passed() is False


def test_mrmr_fit_clears_deadline_after_run():
    """Regression test (2026-08-05): MRMR.fit's outer fit() wrapper sets a deadline (when
    max_runtime_mins is not None) inside _fit_impl but historically NOTHING ever called
    clear_fe_deadline() afterward, despite _fe_deadline.py's own docstring claiming it was "cleared in a
    finally" - the deadline silently persisted as a stale absolute timestamp for the rest of the
    process/thread, so once the timestamp elapsed EVERY later fe_deadline_passed() consumer (including an
    unrelated later MRMR.fit call with max_runtime_mins=None, or any other code on the same thread) saw a
    spuriously-expired budget. Caught live via cross-test flakiness once more FE families started
    consulting fe_deadline_passed(). Pins that a fit with a (already-elapsed, to make the assertion fast)
    max_runtime_mins leaves NO deadline behind once fit() returns."""
    import numpy as np
    import pandas as pd
    from mlframe.feature_selection.filters.mrmr import MRMR

    clear_fe_deadline()
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(5)})
    y = (rng.standard_normal(n) > 0).astype(int)

    # A budget so small it is already exceeded by the time _fit_impl checks it - exercises the
    # set-a-real-deadline path without slowing the test down.
    m = MRMR(max_runtime_mins=1e-9)
    m.fit(X, y)

    assert fe_deadline_passed() is False, "MRMR.fit must clear its FE deadline before returning, not leak it"
    assert fe_budget_active() is False, "MRMR.fit must clear its FE deadline before returning, not leak it"


def test_deadline_isolated_per_thread():
    """The deadline is a THREAD-LOCAL: setting it on the main thread must not be visible on another
    thread (mirrors the module's own documented contract that a future joblib-worker consumer would
    need explicit re-publish, not implicit propagation)."""
    import threading

    set_fe_deadline(timer() - 1.0)
    assert fe_deadline_passed() is True

    seen = {}

    def _worker():
        """Read the deadline state from a fresh thread with no explicit re-publish."""
        seen["passed"] = fe_deadline_passed()
        seen["active"] = fe_budget_active()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert seen["active"] is False, "a fresh thread must not inherit the main thread's deadline"
    assert seen["passed"] is False
