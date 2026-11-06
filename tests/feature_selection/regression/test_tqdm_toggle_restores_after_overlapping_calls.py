"""Overlapping hybrid-selector calls in two threads leave ``TQDM_DISABLE`` as the user had it.

Each call used to save and restore the variable itself. When the first call finished it restored the original value while the second was still
running, and the second then restored the value it had saved mid-way, so ``TQDM_DISABLE=1`` stayed set for the rest of the process.
"""

from __future__ import annotations

import os
import threading

from mlframe.feature_selection import hybrid_selector as hs


def test_overlapping_calls_restore_the_user_value(monkeypatch):
    """Thread A enters, thread B enters, A returns, B returns: the user's value is back once both are done."""
    monkeypatch.setenv("TQDM_DISABLE", "0")
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def slow_a():
        """First overlapping call: it stays inside the wrapped method until B has entered."""
        a_in.set()
        b_in.wait(5)

    def slow_b():
        """Second overlapping call: it leaves only after A has returned."""
        b_in.set()
        a_out.wait(5)

    wrapped_a = hs._tqdm_off_during_call(lambda: slow_a())
    wrapped_b = hs._tqdm_off_during_call(lambda: slow_b())
    ta = threading.Thread(target=wrapped_a)
    tb = threading.Thread(target=wrapped_b)
    ta.start()
    a_in.wait(5)
    tb.start()
    ta.join(5)
    a_out.set()
    tb.join(5)
    assert os.environ.get("TQDM_DISABLE") == "0"
