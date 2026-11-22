"""Audit 2026-11-20 finding 34-F34: BoundedMemo keeps the most recent ``capacity`` values and rebuilds an evicted key.

No test covered the capacity eviction. The memo holds two values here; a third key evicts the least recently used one.
"""

from mlframe.training.composite.discovery._per_base_x import BoundedMemo


def test_capacity_evicts_least_recently_used_and_rebuilds():
    """With capacity 2: touching 'a' keeps it, so inserting 'c' evicts 'b'; 'b' is then rebuilt on the next request."""
    memo = BoundedMemo(capacity=2)
    builds = []

    def make(name):
        """Return a builder that records each build of ``name``."""
        def _build():
            """Record and return the value for ``name``."""
            builds.append(name)
            return name.upper()

        return _build

    assert memo.get_or_build("a", make("a")) == "A"
    assert memo.get_or_build("b", make("b")) == "B"
    assert memo.get_or_build("a", make("a")) == "A"  # hit: 'a' becomes most recent
    assert memo.get_or_build("c", make("c")) == "C"  # evicts 'b', the least recent
    assert memo.get_or_build("b", make("b")) == "B"  # rebuilt
    assert builds == ["a", "b", "c", "b"]
