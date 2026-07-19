"""Silent-degenerate-fallback audit for ``edges_bayesian_blocks`` and ``edges_optimal_joint``
(``_adaptive_nbins.py``). Companion to the MDLP audit that found ``mdlp_bin_edges`` silently
returning EMPTY edges for a long time (``3.0**n_classes`` overflow on a high-cardinality target
made the split-acceptance check always-false). These tests hunt for the SAME bug class in the two
other adaptive-binning strategies: extreme cardinality/skew/tiny-n/degenerate input silently
collapsing to an empty or nonsensical edge array with no signal that anything went wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    edges_bayesian_blocks,
    edges_optimal_joint,
    _bin_y_for_mi,
)

# -----------------------------------------------------------------------------
# edges_bayesian_blocks
# -----------------------------------------------------------------------------


def test_bayesian_blocks_extreme_skew_heavy_tail_not_silently_empty():
    """99% mass at 0 + a heavy exponential tail is exactly the kind of adversarial-but-plausible
    (sensor/financial) column that could trip a T_cp/ncp_prior degeneracy. There IS real structure
    here (two well-separated populations) so BB must not silently collapse to 0 inner edges."""
    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(9900), rng.exponential(1e6, 100)])
    edges = edges_bayesian_blocks(x)
    assert edges.size > 0, "Bayesian Blocks silently returned EMPTY edges on a heavy-tailed column with real structure"
    assert np.all(np.isfinite(edges))


def test_bayesian_blocks_extreme_magnitude_range_finite_edges():
    """Two clusters at 1e-300 and 1e300 -- checks the O(N^2) DP's T_cp/log(N_cp/T_cp) computation
    does not overflow/NaN on an extreme dynamic range."""
    x = np.concatenate([np.full(50, 1e-300), np.full(50, 1e300)])
    edges = edges_bayesian_blocks(x)
    assert edges.size > 0
    assert np.all(np.isfinite(edges)), f"non-finite edge from extreme-magnitude input: {edges}"


@pytest.mark.parametrize("n", [2, 3, 5])
def test_bayesian_blocks_very_small_n_does_not_crash(n):
    """Tiny-n columns (n=2,3,5) must not crash the O(N^2) DP or emit non-finite edges."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    edges = edges_bayesian_blocks(x)
    assert np.all(np.isfinite(edges))


def test_bayesian_blocks_all_identical_returns_empty_not_garbage():
    """A genuinely constant column has no structure to bin -- returning empty edges here is the
    CORRECT degenerate answer (not the bug class under test), but pin it explicitly so a future
    change doesn't silently start emitting NaN/duplicate edges instead of the documented empty-array
    contract for a truly signal-free column."""
    x = np.full(200, 7.0)
    edges = edges_bayesian_blocks(x)
    assert edges.size == 0
    assert np.all(np.isfinite(edges))


def test_bayesian_blocks_large_n_completes_and_is_finite():
    """O(N^2) DP at a few thousand points must complete and stay numerically sane (no perf
    assertion here -- see the module docstring's documented O(N^2) cost note -- just correctness)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3000)
    edges = edges_bayesian_blocks(x)
    assert edges.size > 0
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0), "edges must be strictly increasing"


def test_bayesian_blocks_biz_val_detects_known_step_structure():
    """Biz-val: a clean 3-block piecewise-constant signal (with light noise) has TWO true
    change-points at x=10 and x=20. Bayesian Blocks should place at least 2 inner edges and they
    should land close to the true breakpoints -- not 0 edges (signal lost) and not a degenerate
    single split miles from the truth."""
    rng = np.random.default_rng(42)
    n_per_block = 300
    block0 = rng.normal(0.0, 0.05, n_per_block)
    block1 = rng.normal(10.0, 0.05, n_per_block)
    block2 = rng.normal(20.0, 0.05, n_per_block)
    x = np.concatenate([block0, block1, block2])
    edges = edges_bayesian_blocks(x, p0=0.05)
    assert edges.size >= 2, f"expected >=2 inner edges recovering the 2 true breakpoints, got {edges}"
    # Every true breakpoint (10, 20) should have a recovered edge within 1.0 of it.
    for true_bp in (10.0, 20.0):
        closest = float(np.min(np.abs(edges - true_bp)))
        assert closest < 1.0, f"no recovered edge within 1.0 of true breakpoint {true_bp}: edges={edges}"


# -----------------------------------------------------------------------------
# edges_optimal_joint / _bin_y_for_mi
# -----------------------------------------------------------------------------


def test_bin_y_for_mi_caps_high_cardinality_int_y():
    """Regression test for a real bug: a continuous target mistakenly int64-typed (timestamp,
    counter, upstream float->int cast) was previously treated as ONE DISCRETE CLASS PER DISTINCT
    VALUE with no cardinality check (``K_y = y.max() + 1``). ``_plug_in_mi_njit`` allocates a dense
    ``(K_x, K_y)`` joint-count matrix -- confirmed via a separate-process repro to SEGFAULT the
    interpreter on an int64 target with ~50k unique values (n=50000, oversized np.zeros
    allocation). Same bug class as MDLP's ``max_y_classes`` overflow. Fixed by quantizing
    high-cardinality int/bool y through the same 10-quantile path used for float y."""
    y = np.arange(100_000, dtype=np.int64)  # 100k unique classes -- would have meant K_y=100_000
    y_b, K_y = _bin_y_for_mi(y, max_y_classes=64)
    assert K_y <= 11, f"high-cardinality int y must be quantized (K_y<=11 for 10-quantile binning), got K_y={K_y}"
    assert y_b.size == y.size


def test_bin_y_for_mi_leaves_low_cardinality_int_y_as_class_labels():
    """Low-cardinality int y (genuine classification target) must still be treated as exact class
    labels, not accidentally quantized -- the cardinality cap must not regress the normal case."""
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    y_b, K_y = _bin_y_for_mi(y, max_y_classes=64)
    assert K_y == 3
    np.testing.assert_array_equal(y_b, y)


def test_edges_optimal_joint_high_cardinality_int_y_completes_and_returns_sane_edges():
    """End-to-end: edges_optimal_joint on a continuous-but-int64-typed target must complete quickly
    (not segfault / hang / OOM) and return non-degenerate, finite, sorted edges."""
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.standard_normal(n)
    # Continuous target with a WIDE integer range and near-full cardinality (n-1 unique values) --
    # exactly the "sensor id / timestamp mistakenly int-typed" shape that triggered the bug.
    y_cont = (x * 3.0 + rng.standard_normal(n) * 5.0) * 1_000_000
    y_int = y_cont.astype(np.int64)
    assert np.unique(y_int).size > 1000  # confirms the high-cardinality premise holds
    edges = edges_optimal_joint(x, y_int, candidates=(4, 8, 16, 32))
    assert edges.size > 0
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)


def test_edges_optimal_joint_no_candidate_scored_falls_back_not_empty():
    """Regression test for a real bug: when NO candidate M in ``candidates`` is ever scored by ANY
    CV fold (here: every candidate exceeds every fold's train size except a caller-supplied M=1
    probe), the pre-fix code silently fell back to ``candidates[0]`` UNVALIDATED by the CV search.
    With ``candidates[0] == 1`` this produced completely EMPTY edges (``_edges_from_quantiles``'s
    n_bins<2 guard) with zero signal that the CV search never actually ran -- the exact MDLP
    silent-empty-output bug class. Fixed: fall back to Freedman-Diaconis (the same fallback used
    for the too-small-n guard) when nothing was ever scored."""
    rng = np.random.default_rng(1)
    n = 20
    x = rng.standard_normal(n)
    y = rng.integers(0, 2, n)
    # M=1000 always exceeds n_train (~13-14 per fold at n_splits=3); M=1 is filtered out of the
    # scoring dict entirely (M<2 guard) -- so nothing is ever scored for either candidate.
    edges = edges_optimal_joint(x, y, candidates=(1, 1000), n_splits=3)
    assert edges.size > 0, "CV search that scored nothing must fall back to a real binning, not silent empty edges"
    assert np.all(np.isfinite(edges))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_edges_optimal_joint_very_small_n_does_not_crash(n):
    """Tiny-n (n=2,3,5) must route through the too-small-n FD fallback without crashing."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.integers(0, 2, n)
    edges = edges_optimal_joint(x, y)
    assert np.all(np.isfinite(edges))


def test_edges_optimal_joint_all_identical_x_returns_empty_not_garbage():
    """A constant x column has no bins to find -- empty edges is the CORRECT degenerate answer here
    (pin it so a regression doesn't start emitting NaN edges instead)."""
    x = np.full(100, 3.0)
    y = np.arange(100) % 2
    edges = edges_optimal_joint(x, y)
    assert edges.size == 0
    assert np.all(np.isfinite(edges))


def test_edges_optimal_joint_biz_val_recovers_more_structure_than_noise():
    """Biz-val: on an x with real joint structure with y, CV-selected binning should score higher
    mean fold-MI (more candidates able to resolve signal) than on pure noise -- i.e. the CV
    machinery is actually distinguishing signal from noise, not silently picking garbage either
    way. Measured via downstream MI of the returned binning against y."""
    from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi

    rng = np.random.default_rng(5)
    n = 2000
    x_signal = rng.standard_normal(n)
    y_signal = (x_signal**2 + rng.standard_normal(n) * 0.05 > np.median(x_signal**2)).astype(np.int64)
    x_noise = rng.standard_normal(n)
    y_noise = rng.integers(0, 2, n).astype(np.int64)

    edges_signal = edges_optimal_joint(x_signal, y_signal)
    edges_noise = edges_optimal_joint(x_noise, y_noise)
    assert edges_signal.size > 0

    codes_signal = np.searchsorted(edges_signal, x_signal, side="right").astype(np.int64)
    mi_signal = _plug_in_mi(codes_signal, y_signal)
    if edges_noise.size > 0:
        codes_noise = np.searchsorted(edges_noise, x_noise, side="right").astype(np.int64)
        mi_noise = _plug_in_mi(codes_noise, y_noise)
    else:
        mi_noise = 0.0
    assert mi_signal > 0.15, f"expected clear XOR-like signal to be picked up by CV-selected binning, got MI={mi_signal}"
    assert mi_signal > mi_noise + 0.1, f"signal MI ({mi_signal}) should clearly exceed noise MI ({mi_noise})"
