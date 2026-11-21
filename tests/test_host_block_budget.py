"""The host block budget admits steps while the bytes in flight stay within it, makes the others wait, admits an oversized request alone and never deadlocks a thread that already holds a reservation."""

import threading
import time

from mlframe.utils.host_block_budget import HostBlockBudget


def test_steps_within_the_budget_run_together_and_the_rest_wait():
    """Budget of 3 units, five steps of 1 unit: never more than three in flight, all five finish."""
    b = HostBlockBudget(3)
    now = [0]
    peak = [0]
    lock = threading.Lock()

    def step():
        """One step holding one unit for a moment."""
        with b.reserve(1):
            with lock:
                now[0] += 1
                peak[0] = max(peak[0], now[0])
            time.sleep(0.15)
            with lock:
                now[0] -= 1

    threads = [threading.Thread(target=step) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 3 and b.in_flight() == 0 and b.waits >= 1


def test_an_oversized_request_is_admitted_alone():
    """A request above the whole budget does not wait for ever: it runs when nothing else holds a reservation."""
    b = HostBlockBudget(10)
    with b.reserve(50):
        assert b.in_flight() == 50
    assert b.in_flight() == 0


def test_a_thread_holding_a_reservation_is_not_blocked_by_a_nested_request():
    """Nested use in one thread cannot deadlock even when the sum exceeds the budget."""
    b = HostBlockBudget(10)
    with b.reserve(8):
        with b.reserve(8):
            assert b.in_flight() == 16
    assert b.in_flight() == 0


def test_release_on_exception_and_unlimited_budget():
    """An exception inside the block still frees the bytes; a budget of 0 limits nothing."""
    b = HostBlockBudget(10)
    try:
        with b.reserve(7):
            raise ValueError("boom")
    except ValueError:
        pass
    assert b.in_flight() == 0
    free = HostBlockBudget(0)
    with free.reserve(10**12):
        assert free.in_flight() == 0
