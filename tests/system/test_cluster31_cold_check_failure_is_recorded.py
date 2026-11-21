"""Audit 2026-11-20 finding 31-F6: a failed cold-cache check records the keys as checked, so later fits do not start a new check each time.

Before the fix a failing check returned before the keys were recorded, so every top-level fit started another daemon thread that reloaded
the cache. The test makes the check raise and checks the registry keys are recorded as checked.
"""

from mlframe.system.kernel_tuning_cache import _policy


def test_failed_cold_check_records_the_keys_as_checked(monkeypatch):
    """With cold_kernel_names raising, _cold_check sets _CHECKED_KEYS to the current registry keys."""

    def failing_names():
        """Simulate a corrupt per-host cache file."""
        raise OSError("corrupt cache")

    monkeypatch.setattr(_policy, "cold_kernel_names", failing_names)
    monkeypatch.setattr(_policy, "get_registry_keys", lambda: ["k1", "k2"])
    monkeypatch.setattr(_policy, "_CHECKED_KEYS", frozenset())
    _policy._cold_check()
    assert _policy._CHECKED_KEYS == frozenset({"k1", "k2"})
