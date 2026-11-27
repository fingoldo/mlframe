"""Cluster 28 regression: the get_training_configs memo state lives only in ``_trainer_configure`` (audit 2026-11-20, 28-F47)."""

from mlframe.training import _trainer_configure as live
from mlframe.training import _trainer_configure_helpers as helpers


def test_live_module_owns_the_cache_lock_and_pin():
    """The cache, its lock, its pin dict and the cap exist on the live module (28-F47)."""
    for name in ("_GTC_CACHE", "_GTC_CACHE_LOCK", "_GTC_CACHE_SUBGROUPS_PIN", "_GTC_CACHE_MAX"):
        assert hasattr(live, name), name


def test_helpers_module_no_longer_carries_a_dead_copy():
    """The never-read duplicate in the helpers module is gone, so it cannot be cleared or inspected by mistake (28-F47)."""
    for name in ("_GTC_CACHE", "_GTC_CACHE_LOCK", "_GTC_CACHE_SUBGROUPS_PIN", "_GTC_CACHE_MAX"):
        assert not hasattr(helpers, name), name
