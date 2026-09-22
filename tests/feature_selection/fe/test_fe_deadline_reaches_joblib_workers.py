"""The deadline wrapper publishes the dispatching thread's budget inside the worker that runs the payload.

``fe_deadline_passed`` reads a thread-local. A joblib worker runs on another thread, or in another process, so a payload that checks the
deadline there sees it unset and runs to completion no matter how long the budget was. The wrapper captures the value where the thread-local
actually lives and republishes it around the call.
"""

from __future__ import annotations

import time

import pytest

from mlframe.feature_selection.filters._fe_deadline import (
    DeadlineCarrying,
    clear_fe_deadline,
    current_fe_deadline,
    fe_deadline_passed,
    set_fe_deadline,
)


@pytest.fixture(autouse=True)
def _no_leaked_deadline():
    """A deadline must never leak out of a test and into the rest of the session."""
    clear_fe_deadline()
    yield
    clear_fe_deadline()


def _probe(_ignored):
    """Report whether the deadline is visible wherever this runs."""
    return fe_deadline_passed()


def test_a_passed_deadline_is_visible_inside_a_thread_worker():
    """Dispatched under a thread backend, the payload must see the budget the main thread published."""
    from joblib import Parallel, delayed

    set_fe_deadline(time.perf_counter() - 1.0)  # already elapsed
    got = Parallel(n_jobs=2, backend="threading")(delayed(DeadlineCarrying(_probe))(i) for i in range(4))
    assert all(got), f"the worker did not see the passed deadline: {got}"


def test_without_the_wrapper_the_worker_is_blind_to_the_deadline():
    """The reason the wrapper exists: a bare payload on a fresh thread reads the thread-local as unset."""
    from joblib import Parallel, delayed

    set_fe_deadline(time.perf_counter() - 1.0)
    got = Parallel(n_jobs=2, backend="threading")(delayed(_probe)(i) for i in range(4))
    assert not any(got), f"this fixture is meant to demonstrate the blind spot, but the workers saw the deadline: {got}"


def test_a_future_deadline_reads_as_not_passed_in_the_worker():
    """Carrying the budget must carry its meaning too, not just the fact that one exists."""
    from joblib import Parallel, delayed

    set_fe_deadline(time.perf_counter() + 600.0)
    got = Parallel(n_jobs=2, backend="threading")(delayed(DeadlineCarrying(_probe))(i) for i in range(4))
    assert not any(got), f"a budget 10 minutes away read as spent: {got}"


def test_the_wrapper_captures_at_dispatch_not_at_call():
    """The value is read on the dispatching thread; reading it inside the worker would find the wrong thread-local."""
    set_fe_deadline(1234.5)
    carried = DeadlineCarrying(_probe)
    assert carried.deadline == 1234.5
    clear_fe_deadline()
    assert current_fe_deadline() is None
    assert carried.deadline == 1234.5, "the wrapper re-read the thread-local instead of keeping what it captured"


def test_the_worker_deadline_does_not_outlive_the_call():
    """loky reuses workers, so a deadline written in one and left there would truncate later, budget-free fits."""
    set_fe_deadline(999.0)
    carried = DeadlineCarrying(lambda _x: current_fe_deadline())
    clear_fe_deadline()
    inside = carried(None)
    assert inside == 999.0
    assert current_fe_deadline() is None, "the wrapper left its deadline published after the call returned"
