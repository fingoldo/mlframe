"""Cluster 39 regression: derived memory budgets stay out of the user's sub-configs, so a dumped config re-derives where it is loaded (audit 2026-11-20, 39-F28)."""

from unittest import mock

from mlframe.training.feature_handling.config import CacheConfig, FeatureHandlingConfig, MemoryConfig


def _machine(total_gb):
    """Patch the memory probe to report a machine of the given size."""
    vm = mock.patch("mlframe.training.feature_handling.system.psutil.virtual_memory", return_value=mock.MagicMock(total=int(total_gb * 1e9)))
    cg = mock.patch("mlframe.training.feature_handling.system._read_cgroup_memory_limit_bytes", return_value=None)
    return vm, cg


def test_user_fields_stay_unset_and_the_resolved_views_are_filled():
    """With nothing set, memory.budget_gb / cache.ram_* stay None; resolved_memory and resolved_cache carry the derived numbers (39-F28)."""
    vm, cg = _machine(64)
    with vm, cg:
        fhc = FeatureHandlingConfig()
    assert fhc.memory.budget_gb is None and fhc.cache.ram_max_gb is None and fhc.cache.ram_reserve_gb is None
    assert fhc.resolved_memory.budget_gb == fhc.resolved["memory_budget_gb"] > 0
    assert fhc.resolved_cache.ram_max_gb == fhc.resolved["cache_ram_max_gb"] > 0
    assert fhc.resolved_cache.ram_reserve_gb == fhc.resolved["cache_ram_reserve_gb"] > 0


def test_dump_from_a_big_machine_loads_on_a_small_one_and_caps():
    """A config dumped on 256 GB re-validates on 8 GB by deriving again, instead of raising the explicit-ram_max error (39-F28)."""
    vm, cg = _machine(256)
    with vm, cg:
        big = FeatureHandlingConfig()
    dumped = big.model_dump()
    vm, cg = _machine(8)
    with vm, cg:
        small = FeatureHandlingConfig(**dumped)
    assert small.resolved_cache.ram_max_gb < big.resolved_cache.ram_max_gb
    assert small.resolved_memory.budget_gb < big.resolved_memory.budget_gb


def test_dump_from_a_small_machine_does_not_pin_the_small_budget():
    """A config dumped on 8 GB and loaded on 256 GB gets the large budget, not the small one it was dumped with (39-F28)."""
    vm, cg = _machine(8)
    with vm, cg:
        small = FeatureHandlingConfig()
    vm, cg = _machine(256)
    with vm, cg:
        big = FeatureHandlingConfig(**small.model_dump())
    assert big.resolved_memory.budget_gb > small.resolved_memory.budget_gb * 10


def test_explicit_values_are_kept_and_still_validated():
    """An explicit budget and cache cap come back unchanged from both the field and the resolved view (39-F28)."""
    vm, cg = _machine(64)
    with vm, cg:
        fhc = FeatureHandlingConfig(memory=MemoryConfig(budget_gb=10.0), cache=CacheConfig(ram_max_gb=2.0))
    assert fhc.memory.budget_gb == 10.0 and fhc.resolved_memory.budget_gb == 10.0
    assert fhc.cache.ram_max_gb == 2.0 and fhc.resolved_cache.ram_max_gb == 2.0
