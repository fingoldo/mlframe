"""A deadline republished inside a worker must not outlive the call that published it.

`_fe_deadline`'s state is a `threading.local`, which crosses neither the loky process boundary nor the
big-stack sub-thread the polynom-pair search runs on, so `_eval_one_pair_impl` re-publishes the caller's
absolute deadline into its own execution context. It did so with a bare `set_fe_deadline` and no matching
clear on any exit path.

loky REUSES its worker processes across `Parallel(...)` invocations. The absolute `timer()` timestamp
therefore survived in the worker after the function returned, and once it elapsed every
`fe_deadline_passed()` consumer running in that worker returned True for the rest of its life -- across
later FE steps and later `MRMR.fit()` calls, including ones given no budget at all. The `finally` in
`MRMR.fit` clears the main thread's copy and cannot reach a worker process.

The tests below model that reuse directly: one thread runs the worker body twice, the second time with no
budget, and the second run must not be truncated.
"""

from __future__ import annotations

from contextlib import nullcontext
from timeit import default_timer as timer

import pytest

from mlframe.feature_selection.filters._fe_deadline import (
    clear_fe_deadline,
    fe_budget_active,
    fe_deadline_passed,
    fe_deadline_scope,
    set_fe_deadline,
)

ITERATIONS = 5


@pytest.fixture(autouse=True)
def _no_deadline_left_behind():
    """Every test starts and finishes with the thread-local unset."""
    clear_fe_deadline()
    yield
    clear_fe_deadline()


def _worker_body(deadline) -> int:
    """The shape of `_eval_one_pair_impl`: republish the caller's deadline, then run a bounded search loop."""
    completed = 0
    budget = fe_deadline_scope(deadline) if deadline is not None else nullcontext()
    with budget:
        for _ in range(ITERATIONS):
            if fe_deadline_passed():
                break
            completed += 1
    return completed


def _leaky_worker_body(deadline) -> int:
    """The pre-fix shape: publish and never clear. Kept so the tests can show what they are guarding against."""
    completed = 0
    if deadline is not None:
        set_fe_deadline(deadline)
    for _ in range(ITERATIONS):
        if fe_deadline_passed():
            break
        completed += 1
    return completed


def test_an_expired_deadline_does_not_truncate_the_next_call():
    """The reuse case: a budgeted call followed by an unbudgeted one on the same thread."""
    assert _worker_body(timer() - 1.0) == 0, "an already-expired budget should stop the loop immediately"
    assert _worker_body(None) == ITERATIONS, "the next call, given no budget, was truncated by the previous one's deadline"


def test_the_pre_fix_shape_does_leak():
    """Pins what the fix is for: the same two calls through the old idiom truncate the second one.

    Without this the test above could pass for the wrong reason -- a fixture too fast to reach any deadline,
    or a loop that never checks. Here the leak is reproduced deliberately, on the same clock.
    """
    assert _leaky_worker_body(timer() - 1.0) == 0
    assert _leaky_worker_body(None) == 0, "the fixture no longer reproduces the leak it was built to show"


def test_the_scope_restores_an_outer_budget_rather_than_clearing_it():
    """Restoring, not clearing: a nested publisher must not wipe the budget its caller is running under."""
    outer = timer() + 3600.0
    set_fe_deadline(outer)
    with fe_deadline_scope(timer() - 1.0):
        assert fe_deadline_passed() is True
    assert fe_budget_active() is True, "the outer budget was dropped by the nested scope"
    assert fe_deadline_passed() is False, "the outer budget came back expired"


def test_the_scope_restores_on_an_exception():
    """A worker that raises must not leave its deadline behind either; loky reuses the process regardless."""
    with pytest.raises(RuntimeError):
        with fe_deadline_scope(timer() - 1.0):
            raise RuntimeError("the search blew up")
    assert fe_budget_active() is False, "a raising worker left its deadline in the thread-local"


def test_the_scope_leaves_nothing_behind_when_nothing_was_set():
    """The common path: no budget before, none after."""
    assert fe_budget_active() is False
    with fe_deadline_scope(timer() + 3600.0):
        assert fe_budget_active() is True
    assert fe_budget_active() is False
