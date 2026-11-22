"""Audit 2026-11-20 finding 34-F28: BoundedMemo runs the build outside its lock, so a re-entrant build cannot deadlock.

Before the fix the build ran under the lock, and a build that asked the same memo for another key blocked forever.
"""

import threading

from mlframe.training.composite.discovery._per_base_x import BoundedMemo


def test_reentrant_build_does_not_deadlock():
    """A build that reads another key from the same memo returns instead of hanging."""
    memo = BoundedMemo(capacity=4)
    result = {}

    def outer():
        """Build 'a' by first building 'b' from the same memo."""
        result["a"] = memo.get_or_build("a", lambda: ("a", memo.get_or_build("b", lambda: "b")))

    t = threading.Thread(target=outer, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "re-entrant build deadlocked"
    assert result["a"] == ("a", "b")


def test_repeated_key_is_built_once_when_sequential():
    """Sequential calls for one key build once and reuse the value."""
    memo = BoundedMemo(capacity=2)
    calls = []
    memo.get_or_build("k", lambda: calls.append(1) or 7)
    assert memo.get_or_build("k", lambda: calls.append(1) or 8) == 7
    assert len(calls) == 1
