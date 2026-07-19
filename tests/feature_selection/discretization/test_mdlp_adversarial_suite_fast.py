"""Fast smoke subset for the MDLP validated-split adversarial suite (2026-07-19).

Runs every scenario in ``bench_mdlp_adversarial_suite.py`` at a tiny seed count (a few seconds
total) as a real pytest -- exercises the near-boundary noise stress, the real-MRMR confounder
redundancy check, the extreme-imbalance/outlier stress, and the multi-comparisons-defeat harness
without paying the full 30-50-seed cost (that lives in the module's ``__main__``, run on demand,
not in CI).
"""
from __future__ import annotations

import math

from mlframe.feature_selection.filters._benchmarks.bench_mdlp_adversarial_suite import (
    run_extreme_imbalance_boundary_stress,
    run_extreme_outlier_boundary_stress,
    run_mrmr_confounder_redundancy_stress,
    run_multi_comparisons_defeat,
    run_near_boundary_noise_stress,
)


def test_near_boundary_noise_stress_runs_and_stays_bounded():
    """Every cell must produce a finite accept rate in [0, 1]; the whole sweep must run without
    raising across both the analytic and permutation code paths (the sweep straddles the
    ``analytic_null_min_n()`` floor by construction)."""
    results = run_near_boundary_noise_stress(n_seeds=3)
    assert results
    for r in results:
        assert 0.0 <= r.rate <= 1.0
        assert math.isfinite(r.ci_lo) and math.isfinite(r.ci_hi)


def test_mrmr_confounder_redundancy_stress_runs():
    """Real MRMR.fit on the pure-confounder scenario must run without raising and return a
    boolean selection decision for both the relevant and confounder columns, every seed."""
    results = run_mrmr_confounder_redundancy_stress(n_seeds=3, n=1200)
    assert len(results) == 3
    for r in results:
        assert isinstance(r.relevant_selected, bool)
        assert isinstance(r.confounder_selected, bool)


def test_extreme_imbalance_boundary_stress_no_crashes():
    """No cell may report a crash (any exception during the fit) -- a crash under stress is a
    real production bug, not an acceptable outcome of an adversarial scenario."""
    results = run_extreme_imbalance_boundary_stress(n_seeds=3)
    assert results
    for r in results:
        assert r.n_crashes == 0, f"crash(es) at n={r.n} minority_frac={r.minority_frac}"


def test_extreme_outlier_boundary_stress_no_crashes_no_nonfinite_edges():
    """Extreme-outlier continuous targets must not crash the binner and must not produce
    non-finite INNER edges (the -inf/+inf sentinels are expected and excluded)."""
    results = run_extreme_outlier_boundary_stress(n_seeds=3)
    assert results
    for r in results:
        assert r.n_crashes == 0, f"crash(es) at n={r.n} outlier_frac={r.outlier_frac}"
        assert r.n_nonfinite == 0, f"non-finite inner edge(s) at n={r.n} outlier_frac={r.outlier_frac}"


def test_multi_comparisons_defeat_runs_and_reports_finite_fdr():
    """The tree-wide FDR sweep must run across every (n, n_classes, bonferroni) cell without
    raising and report a finite rate in [0, 1]."""
    results = run_multi_comparisons_defeat(n_seeds=3)
    assert results
    for r in results:
        assert 0.0 <= r.tree_fdr <= 1.0
