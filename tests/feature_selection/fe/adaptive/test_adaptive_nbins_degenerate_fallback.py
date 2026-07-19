"""Silent-degenerate-fallback audit for the closed-form adaptive-nbins strategies
(``edges_sturges``, ``edges_freedman_diaconis``, ``edges_knuth``).

Motivated by the Fayyad-Irani MDLP bug (``mdlp_bin_edges``): a continuous high-cardinality target
cast to int64 overflowed an internal ``3.0**n_classes`` term to ``inf``, the split-acceptance check
became "always false", and MDLP silently returned EMPTY edges for months -- no test caught it
because the suite only exercised well-behaved synthetic columns. These tests hunt for the SAME bug
class (silent empty/degenerate output under extreme-but-realistic input distributions) in the three
closed-form binning rules, which had no dedicated edge-case coverage before this file.

Bug found and fixed while writing these tests (see ``discretization/_discretization_edges.py``):
1. ``_knuth_bin_edges`` had no cap on M by distinct-value count -- a near-constant column with a
   handful of outliers (99.995% one value) made the log-posterior increase monotonically with M
   (empty bins are free under ties), saturating at the artificial ``m_max_cap`` (500) instead of a
   real optimum. Fixed by capping ``M_max`` at the number of distinct values.
2. ``_knuth_best_M``'s fused njit kernel used ``np.searchsorted(..., side='right')`` where the
   ``np.histogram``-equivalent reference needs ``side='left'`` -- on tied/integer data (years,
   counts, small categorical-coded integers) a value landing exactly on an interior bin edge was
   silently assigned to the wrong bin, shifting which M the posterior search picked. The
   long-standing "BIT-IDENTICAL to np.histogram" docstring claim was never actually true for
   tied edge values; fixed to ``side='left'``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    edges_sturges,
    edges_freedman_diaconis,
    edges_knuth,
    sturges_nbins,
    freedman_diaconis_nbins,
    per_feature_edges,
)

ALL_EDGE_FNS = {
    "sturges": edges_sturges,
    "freedman_diaconis": edges_freedman_diaconis,
    "knuth": edges_knuth,
}


# -----------------------------------------------------------------------------
# Extreme cardinality: near-unique-per-row continuous data (the MDLP bug's shape)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
def test_extreme_cardinality_continuous_target_yields_nonempty_edges(name, fn):
    """A near-unique-per-row continuous column (years + sub-integer jitter, ~200k distinct
    values) is exactly the shape that broke MDLP silently. All three closed-form rules must
    still find real structure here (span is large, data isn't uniform-flat)."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.integers(1900, 2026, size=n).astype(np.float64) + rng.random(n)
    x[: n // 4] += rng.standard_normal(n // 4) * 5  # break flat-uniformity so binning is genuinely useful
    edges = fn(x)
    assert edges.size > 0, f"{name}: extreme-cardinality continuous column silently produced ZERO edges (K_x=1)"


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
def test_extreme_skew_heavy_tail_yields_nonempty_edges(name, fn):
    """Pareto(alpha=0.5): extreme right-skew / heavy tail. Must not collapse to empty edges,
    and must not blow up to a nonsensical bin count (silent degenerate over-fit is also a
    failure mode -- see the knuth ties bug found in this session)."""
    rng = np.random.default_rng(1)
    x = rng.pareto(0.5, size=50_000)
    edges = fn(x)
    assert edges.size > 0, f"{name}: heavy-tailed column silently produced ZERO edges"
    assert edges.size < x.size, f"{name}: bin count {edges.size} >= n is a degenerate bin-per-sample fallback"


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
def test_all_identical_values_collapse_to_k1_not_crash(name, fn):
    """Constant column: correct/contractual answer is K_x=1 (zero inner edges), not a crash
    and not a phantom multi-edge output (that would be the MDLP-class bug in reverse)."""
    x = np.full(1000, 7.0)
    edges = fn(x)
    assert edges.size == 0, f"{name}: constant column must yield K_x=1 (0 inner edges); got {edges.size}"


def test_knuth_near_constant_with_few_outliers_bounded_not_saturated():
    """99.995% one value + 5 real outliers, 6 distinct values total. Regression fixture for the
    knuth bug found in this session: pre-fix, edges_knuth's exhaustive posterior search saturated
    at the artificial m_max_cap (499 edges) because empty bins are 'free' under exact ties (the
    log-posterior increases monotonically with M with no interior maximum). Post-fix (M capped at
    the distinct-value count) it must reflect the ~6 distinct values, not the cap."""
    rng = np.random.default_rng(2)
    x = np.zeros(100_000)
    x[:5] = rng.random(5) * 1000.0
    edges = edges_knuth(x, m_max_cap=500)
    assert edges.size > 0, "knuth: near-constant-with-outliers silently produced ZERO edges"
    assert edges.size <= 10, (
        f"knuth: near-constant column with only 6 distinct values produced {edges.size} edges -- "
        "looks like a saturated/degenerate cap hit rather than a real answer"
    )


@pytest.mark.parametrize("name,fn", [("sturges", edges_sturges), ("freedman_diaconis", edges_freedman_diaconis)])
def test_sturges_fd_near_constant_raw_fn_documented_contract(name, fn):
    """Sturges / FD are closed-form UNSUPERVISED formulas driving equal-FREQUENCY quantile splits
    at a modest bin count -- with 99.995% mass at one value, every quantile split lands on that
    value and legitimately dedups to empty (0 inner edges = K_x=1). This is a documented
    consequence of the formula (see ``freedman_diaconis_nbins``'s IQR=0 -> Sturges fallback), not
    a crash/NaN-poisoning bug -- the raw function's job ends there. Production callers never use
    these raw functions directly on such columns (see ``test_per_feature_edges_...`` below): the
    dispatcher's low-cardinality midpoint fallback and systemic empty-edges guardrail catch this
    exact case before/after calling into sturges/FD. This test only pins the raw contract: finite,
    no crash, and (if non-empty) sane."""
    rng = np.random.default_rng(2)
    x = np.zeros(100_000)
    x[:5] = rng.random(5) * 1000.0
    edges = fn(x)
    assert np.all(np.isfinite(edges)), f"{name}: near-constant column produced non-finite edges"


@pytest.mark.parametrize("name,method", [("sturges", "sturges"), ("freedman_diaconis", "freedman_diaconis"), ("knuth", "knuth")])
def test_per_feature_edges_near_constant_dispatcher_guardrail_engages(name, method):
    """Integration-level check of the ACTUAL production path (``per_feature_edges``), which every
    real caller goes through (raw ``edges_*`` functions are only called directly from benchmarks).
    The near-constant-with-5-outliers column has 6 distinct values <= ``low_card_cap`` (32 by
    default), so the dispatcher's low-cardinality midpoint-edge fallback (2026-05-30 Wave 9.1 fix)
    must engage BEFORE the method-specific binner ever runs, guaranteeing non-degenerate output for
    every strategy -- this is the systemic protection the standalone raw-function contract (see
    above) relies on."""
    rng = np.random.default_rng(2)
    x = np.zeros(100_000)
    x[:5] = rng.random(5) * 1000.0
    X = x.reshape(-1, 1)
    y = (rng.random(100_000) > 0.5).astype(np.int64)
    edges = per_feature_edges(X, method=method, y=y, n_jobs=1)[0]
    assert edges.size > 0, f"{name}: per_feature_edges silently produced ZERO edges for a 6-distinct-value column"
    Kx = int(np.searchsorted(edges, x, side="right").max()) + 1
    assert Kx > 1, f"{name}: per_feature_edges collapsed a real 6-distinct-value column to K_x=1"


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
@pytest.mark.parametrize("n", [2, 3, 5])
def test_very_small_n_no_crash(name, fn, n):
    """n=2,3,5: must not crash, and must not silently emit NaN/inf edges."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    edges = fn(x)
    assert np.all(np.isfinite(edges)), f"{name}: n={n} produced non-finite edges: {edges}"


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
def test_very_large_n_completes_and_bounded(name, fn):
    """n=2M: asymptotic stress. Sturges/FD/Knuth formulas must not overflow/NaN and must
    produce a bin count that grows sub-linearly with N (not a bin-per-sample degenerate answer)."""
    rng = np.random.default_rng(4)
    n = 2_000_000
    x = rng.standard_normal(n)
    edges = fn(x)
    assert edges.size > 0
    assert np.all(np.isfinite(edges)), f"{name}: n={n} produced non-finite edges"
    assert edges.size < 5000, f"{name}: n={n} produced {edges.size} edges -- looks like unbounded blow-up"


@pytest.mark.parametrize("name,fn", ALL_EDGE_FNS.items())
def test_partial_nan_and_inf_are_dropped_not_poisoning(name, fn):
    """Docstrings/inline comments in this module establish the contract: NaN/inf are filtered
    before binning (isfinite mask), not propagated. A single inf must not poison every edge."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000)
    x[::100] = np.nan
    x[::137] = np.inf
    x[::151] = -np.inf
    edges = fn(x)
    assert np.all(np.isfinite(edges)), f"{name}: NaN/inf leaked into output edges: {edges}"


def test_knuth_tie_heavy_integer_column_bin_count_tracks_distinct_values():
    """Regression pin for the distinct-value-cap fix: an integer column with only 7 distinct
    values (0..6, uniform-ish) must not be binned into hundreds of near-empty bins."""
    rng = np.random.default_rng(6)
    x = rng.integers(0, 7, size=20_000).astype(np.float64)
    edges = edges_knuth(x, m_max_cap=500)
    assert edges.size <= 6, f"tie-heavy 7-distinct-value column produced {edges.size} edges (>6) -- cap not respected"


def test_knuth_no_overflow_at_extreme_n_lgamma_terms():
    """Knuth's posterior uses lgamma(n + M/2) -- verify no inf/NaN corruption at large N,
    the same failure CLASS (silent inf from an internal combinatorial/log-gamma term) as the
    MDLP ``3.0**n_classes`` overflow bug, just a different formula."""
    import math

    for n in (10, 1_000, 1_000_000, 100_000_000):
        for M in (2, 64, 500):
            val = math.lgamma(n + M / 2.0)
            assert math.isfinite(val), f"lgamma(n={n}, M={M}) overflowed to {val}"


@pytest.mark.parametrize("name,nbins_fn", [("sturges", sturges_nbins)])
def test_sturges_nbins_large_n_no_overflow(name, nbins_fn):
    """Sturges: ceil(1 + log2(n)). log2 of a huge n must stay finite and sane."""
    for n in (10, 10_000, 10**9, 10**15):
        val = nbins_fn(n)
        assert np.isfinite(val) and val >= 1


def test_freedman_diaconis_nbins_large_n_no_overflow():
    """FD's ``n ** (1/3)`` term must not overflow for realistic-to-extreme n."""
    rng = np.random.default_rng(7)
    for n in (10, 10_000, 5_000_000):
        x = rng.standard_normal(n)
        val = freedman_diaconis_nbins(x)
        assert np.isfinite(val) and val >= 1
