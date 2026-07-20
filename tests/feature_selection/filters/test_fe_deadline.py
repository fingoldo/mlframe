"""Direct unit coverage for ``_fe_deadline`` (mrmr_audit_2026-07-20 test_coverage.md #3 /
edge_cases.md #9-12). Only exercised transitively via full MRMR fits before this file -- pins the
thread-local deadline contract directly: no-leak across fits, no cross-thread leak, the no-budget
path never gates, and a 0.0/negative deadline is treated as ALREADY ELAPSED (not falsy-unset).
"""

from __future__ import annotations

import threading
from timeit import default_timer as timer

from mlframe.feature_selection.filters._fe_deadline import (
    clear_fe_deadline,
    fe_budget_active,
    fe_deadline_passed,
    set_fe_deadline,
)


class TestNoDeadlineNeverGates:
    """With no deadline set (the common, no-budget path), fe_deadline_passed() must be False and
    fe_budget_active() must be False -- the enrichment generators run to completion."""

    def test_fresh_state_never_gates(self):
        """A never-touched thread (or one after clear_fe_deadline()) sees no gating."""
        clear_fe_deadline()
        assert not fe_deadline_passed()
        assert not fe_budget_active()


class TestClearedDeadlineDoesNotLeak:
    """clear_fe_deadline() must fully reset state -- a deadline set by one fit must not leak into
    the next fit on the same thread."""

    def test_clear_after_set_resets_to_no_gate(self):
        """Set an already-elapsed deadline, then clear it -- gating must turn back off."""
        set_fe_deadline(timer() - 1.0)  # already elapsed
        assert fe_deadline_passed()
        assert fe_budget_active()

        clear_fe_deadline()
        assert not fe_deadline_passed(), "clear_fe_deadline() must fully reset the gate, not leave it tripped"
        assert not fe_budget_active(), "clear_fe_deadline() must fully reset fe_budget_active() too"

    def test_second_fit_without_a_deadline_does_not_inherit_the_first_fits_deadline(self):
        """Simulates: fit #1 sets a deadline and clears it in a finally; fit #2 sets NO deadline
        (max_runtime_mins=None) -- fit #2 must not see fit #1's leftover state."""
        set_fe_deadline(timer() + 999.0)  # fit #1: far-future deadline
        clear_fe_deadline()  # fit #1's finally

        set_fe_deadline(None)  # fit #2: no budget requested
        assert not fe_deadline_passed()
        assert not fe_budget_active()


class TestThreadLocalDoesNotCrossThreads:
    """The deadline is a threading.local() -- setting it on one thread must not be visible from
    another thread."""

    def test_deadline_set_on_main_thread_not_visible_on_worker_thread(self):
        """Set an already-elapsed deadline on the main thread; a worker thread must see NO deadline."""
        set_fe_deadline(timer() - 1.0)
        assert fe_deadline_passed()  # main thread: gated

        worker_saw_gate = {}

        def _worker():
            """Worker that worker."""
            worker_saw_gate["passed"] = fe_deadline_passed()
            worker_saw_gate["active"] = fe_budget_active()

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        assert not worker_saw_gate["passed"], "a worker thread must not see the main thread's deadline (threading.local() must not cross threads)"
        assert not worker_saw_gate["active"]

        clear_fe_deadline()  # don't leak into other tests on this thread


class TestZeroOrNegativeDeadlineTreatedAsElapsed:
    """A deadline of 0.0 (or negative) is a valid absolute timer() value in the distant past and
    must be treated as ALREADY ELAPSED, not as a falsy 'unset' sentinel -- ``fe_deadline_passed``
    must compare against ``is not None``, not truthiness."""

    def test_zero_deadline_is_treated_as_elapsed_not_unset(self):
        """set_fe_deadline(0.0) must gate immediately (0.0 is far in the past relative to timer())."""
        set_fe_deadline(0.0)
        assert fe_deadline_passed(), (
            "B-regression risk: a deadline of exactly 0.0 must be treated as an elapsed absolute " "timer() value, not as a falsy 'no deadline' sentinel."
        )
        assert fe_budget_active(), "fe_budget_active() must also treat 0.0 as 'a budget IS in effect', not falsy-unset"
        clear_fe_deadline()

    def test_negative_deadline_is_treated_as_elapsed(self):
        """A negative deadline value must also gate immediately."""
        set_fe_deadline(-100.0)
        assert fe_deadline_passed()
        assert fe_budget_active()
        clear_fe_deadline()
