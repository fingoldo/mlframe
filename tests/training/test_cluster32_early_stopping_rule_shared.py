"""Cluster 32 regression: the early-stopping rule is one function shared by production and the benchmark (audit 2026-11-20, finding 32-F21)."""

from mlframe.training._benchmarks import bench_early_stopping_rounds_rule as bench
from mlframe.training._helpers_training_configs import resolve_early_stopping_rounds


def test_auto_rule_derives_a_third_of_iterations():
    """0 (the suite default) becomes max(2, iterations // 3) (32-F21)."""
    assert resolve_early_stopping_rounds(0, 3000) == 1000
    assert resolve_early_stopping_rounds(0, 3) == 2


def test_disabled_and_explicit_values_are_untouched():
    """None stays disabled and a positive patience is kept as given (32-F21)."""
    assert resolve_early_stopping_rounds(None, 3000) is None
    assert resolve_early_stopping_rounds(50, 3000) == 50


def test_benchmark_auto_arm_is_the_production_rule():
    """The benchmark's auto baseline equals the production derivation at its iteration cap (32-F21)."""
    assert bench.AUTO_ES == resolve_early_stopping_rounds(0, bench.ITERATIONS)
