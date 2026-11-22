"""Audit 2026-11-20 findings 07-F2 and 07-F4 in the evaluation benchmarks.

07-F2: the rank-based AUC in the bootstrap-coverage bench must still equal sklearn's ROC AUC after the dead sort is removed.
07-F4: the group-leakage bench swallows only the leak it constructs on purpose; any other ValueError propagates.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.evaluation._benchmarks import bench_bootstrap_ci_coverage as coverage
from mlframe.evaluation._benchmarks import bench_group_leakage_guard as leakage


def test_fast_auc_matches_sklearn_with_ties():
    """The rank-based AUC equals roc_auc_score on scores with ties."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    y[0], y[1] = 0, 1
    s = np.round(rng.normal(size=300) + y, 1)
    assert coverage._fast_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_leak_detected_by_construction_is_swallowed():
    """The synthetic leaky dataset raises the leak ValueError, which the bench treats as expected and returns."""
    assert leakage._run_near_duplicate(n_entities=20, copies_per_entity=3, n_features=4, n_splits=3) is None


def test_non_leak_value_error_propagates(monkeypatch):
    """A ValueError that is not a leak report (an argument problem) is raised, not timed over."""

    def bad_arguments(*args, **kwargs):
        """Stand in for the guard with an argument error."""
        raise ValueError("assert_no_group_leakage: unsupported near_duplicate_metric 'manhattan'")

    monkeypatch.setattr(leakage, "assert_no_group_leakage", bad_arguments)
    with pytest.raises(ValueError, match="unsupported"):
        leakage._run_near_duplicate(n_entities=5, copies_per_entity=2, n_features=3, n_splits=2)


def test_coverage_run_refuses_to_report_when_trials_fail(monkeypatch):
    """Trials that all fail raise instead of dividing by zero, and failures above 5% raise instead of biasing coverage."""

    def draw(rng, n):
        """A trivial draw; the bootstrap is what fails in this test."""
        return np.arange(n, dtype=float), np.arange(n, dtype=float)

    def always_fails(*args, **kwargs):
        """Stand in for bootstrap_metric by failing on every call."""
        raise ValueError("bootstrap blew up")

    monkeypatch.setattr(coverage, "bootstrap_metric", always_fails)
    with pytest.raises(RuntimeError, match="bootstrap trials failed"):
        coverage.run_scenario("t", draw, lambda a, b: 0.5, 0.5, seed=1, n_sample=10, n_trials=4, n_bootstrap=5)


def test_stratify_bench_times_the_production_book():
    """The bench's rebuild is the production helper's book: a list of per-class index arrays with sizes and offsets."""
    from mlframe.evaluation._benchmarks import bench_stratify_groups_cache as stratify_bench

    strat = np.array([0, 1, 1, 0, 2, 2, 2], dtype=np.int64)
    groups_list, class_sizes, class_offsets, total_n = stratify_bench._rebuild(strat)
    assert [g.tolist() for g in groups_list] == [[0, 3], [1, 2], [4, 5, 6]]
    assert class_sizes.tolist() == [2, 2, 3] and total_n == 7
    assert class_offsets.tolist() == [0, 2, 4, 7]
