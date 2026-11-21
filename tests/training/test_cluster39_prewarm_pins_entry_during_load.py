"""Audit 2026-11-20 finding 39-F36: a prewarmed provider is held in the strong LRU tier while it loads, so it cannot be collected mid-load."""

from mlframe.training.feature_handling import registry


class _Stub:
    """Provider stand-in that records whether its own entry is in the strong tier when acquire() runs."""

    signature = "sig-cluster39-f36-test"

    def __init__(self):
        self.in_strong_tier_at_acquire = None

    def acquire(self):
        """Record the strong-tier membership at the moment of loading."""
        self.in_strong_tier_at_acquire = self.signature in registry._LRU_HARD

    def release(self):
        """No resources to free."""


def test_prewarm_holds_the_entry_in_the_strong_tier_while_loading():
    """During acquire() the entry is pinned in the strong tier; afterwards the test removes it again."""
    stub = _Stub()
    try:
        registry.prewarm(stub).result(timeout=120)
        assert stub.in_strong_tier_at_acquire is True
    finally:
        registry._LRU_HARD.pop(stub.signature, None)
