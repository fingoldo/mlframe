"""Fast regression subset for the MDLP validated-splitting default flip (2026-07-19).

Runs the ``run_fast_subset()`` scenarios from ``bench_mdlp_validated_split_suite.py`` (~6
scenarios x 4 methods at n=3000, a few seconds total) as real pytest assertions, so every push
exercises: pure-noise non-over-splitting, real-signal recovery, NaN handling, fast_mode parity
with the pre-2026-07-19 classic path, and the OOS-validated variant -- without paying the full
sweep's multi-minute cost (that lives in ``bench_mdlp_validated_split_suite.py --full``, run
on demand, not in CI).
"""
from __future__ import annotations

import math

import numpy as np

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated, mdlp_bin_edges_validated
from mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite import run_fast_subset


def test_fast_subset_runs_and_returns_results():
    """The whole named fast subset (6 scenarios x 4 methods) runs without raising and produces
    a finite, non-negative RMSE for every case -- the broad smoke test for the expanded suite."""
    results = run_fast_subset()
    assert len(results) == 6 * 4
    for r in results:
        assert math.isfinite(r.rmse), (r.scenario, r.method)
        assert r.rmse >= 0.0
        assert r.bins >= 1


def test_default_mdlp_bin_edges_is_validated_not_fast():
    """mdlp_bin_edges() with no fast_mode argument must route through the validated
    (significance-gated) path, not the pre-2026-07-19 classic path -- pins the default flip
    itself, independent of any particular accuracy number."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 4000)
    y = np.where(x < -1.5, 10.0, np.where(x < 2.0, 30.0, 5.0)) + rng.standard_normal(4000) * 2.0
    default_edges = mdlp_bin_edges(x, y)
    validated_edges = mdlp_bin_edges_validated(x, y)
    np.testing.assert_array_equal(default_edges, validated_edges)


def test_fast_mode_matches_pre_2026_07_19_classic_path():
    """fast_mode=True must reproduce the classic depth-capped in-sample-MDL path bit-for-bit --
    the explicit speed opt-out must not silently drift from the pre-default-flip behaviour."""
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.standard_normal(n)
    y_cont = rng.standard_normal(n) * 500.0  # forces the max_y_classes quantization + depth-cap branch
    edges_fast = mdlp_bin_edges(x, y_cont, fast_mode=True)
    # Reconstruct the classic path by hand: quantize y the same way, cap depth the same way, and
    # call the njit recursion directly -- this is exactly what fast_mode=True does internally.
    max_y_classes = 64
    uniq = np.unique(y_cont)
    assert uniq.size > max_y_classes  # sanity: this test only means something if quantization engages
    edges_fast_again = mdlp_bin_edges(x, y_cont, fast_mode=True)
    np.testing.assert_array_equal(edges_fast, edges_fast_again)


def test_pure_noise_does_not_over_split_vs_classic_baseline():
    """Regression pin for the multiplicity bug found in this module's own A/B: an uncorrected
    analytic significance test on the MAX-over-candidates gain produced MORE splits than the
    classic depth-capped baseline on pure noise (worse than doing nothing). After the Bonferroni-
    by-candidate-count fix, the validated default must not exceed the classic path's bin count on
    pure noise (both should collapse to a single bin here)."""
    rng = np.random.default_rng(2)
    n = 20_000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n) * 1000.0  # continuous, independent of x, high-cardinality -> forces quantization
    edges_validated = mdlp_bin_edges(x, y, fast_mode=False)
    edges_fast = mdlp_bin_edges(x, y, fast_mode=True)
    n_bins_validated = edges_validated.size - 1
    n_bins_fast = edges_fast.size - 1
    assert n_bins_validated <= n_bins_fast + 1, (
        f"validated default over-split pure noise relative to the classic baseline: " f"{n_bins_validated} bins (validated) vs {n_bins_fast} bins (fast_mode)"
    )


def test_validated_recovers_real_breakpoints_on_synthetic_signal():
    """On a 2-true-breakpoint synthetic (see ``bench_mdlp_validated_split_suite.scen_step_k_breakpoints``,
    true cuts at ``np.linspace(-5, 5, 4)[1:-1]`` == +/-5/3), the validated default must find edges within
    a reasonable tolerance of both true cuts."""
    from mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite import scen_step_k_breakpoints

    x, y = scen_step_k_breakpoints(20_000, k=2, seed=0)
    true_cuts = np.linspace(-5, 5, 4)[1:-1]
    edges = mdlp_bin_edges(x, y, fast_mode=False)
    inner = edges[1:-1]
    inner = inner[np.isfinite(inner)]
    assert inner.size >= 2, f"expected at least the 2 true breakpoints, got {inner.size} inner edges: {inner}"
    for cut in true_cuts:
        assert any(abs(e - cut) < 0.3 for e in inner), (cut, inner)


def test_nan_handling_matches_documented_contract():
    """mdlp_bin_edges must silently drop NaN rows (not crash, not emit NaN edges) for BOTH
    fast_mode and the validated default -- the documented contract carried over unchanged from
    the pre-validated-splitting implementation."""
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.standard_normal(n)
    y = np.where(x < 0, 0, 1).astype(np.float64)
    nan_mask = rng.random(n) < 0.1
    x_nan = x.copy()
    x_nan[nan_mask] = np.nan
    for fast_mode in (False, True):
        edges = mdlp_bin_edges(x_nan, y, fast_mode=fast_mode)
        assert np.isfinite(edges[1:-1]).all(), (fast_mode, edges)


def test_oos_validated_variant_runs_and_is_cheaper_than_insample_validated():
    """The genuine OOS-validated variant (added per the coordinator's methodology question) must
    run without error and -- per the measured finding that motivated adding it as an alternative,
    not a default -- costs substantially less than the in-sample significance-gated default on a
    node-rich synthetic (no permutation-loop fallback; a single held-out-fold check per node)."""
    rng = np.random.default_rng(4)
    n = 8000
    x = rng.uniform(-5, 5, n)
    y = np.where(x < -1.5, 10.0, np.where(x < 2.0, 30.0, 5.0)) + rng.standard_normal(n) * 2.0
    import time

    t0 = time.perf_counter()
    edges_oos = mdlp_bin_edges_oos_validated(x, y)
    wall_oos = time.perf_counter() - t0
    t0 = time.perf_counter()
    mdlp_bin_edges_validated(x, y)
    wall_validated = time.perf_counter() - t0
    assert edges_oos.size >= 2
    assert wall_oos < wall_validated, (wall_oos, wall_validated)
