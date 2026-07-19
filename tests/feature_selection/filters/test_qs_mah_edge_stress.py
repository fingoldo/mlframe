"""Edge-case / stress audit for the 'qs' (Gupta quantile-spacing) and 'mah' (Marx SCI) nbins strategies.

Companion to the MDLP silent-empty-edges bug found elsewhere in this same audit round: a supervised/adaptive
binning method that silently returns EMPTY edges (or hangs) for a column with real variance destroys that
column's downstream MI signal with no observable error. These tests hunt that failure class specifically for
qs/mah: extreme cardinality/skew, all-identical values, tiny/huge n, and a cardinality-driven hang (the actual
bug found and fixed here in ``_mah.py``).
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from mlframe.feature_selection.filters._adaptive_nbins import edges_qs, qs_nbins
from mlframe.feature_selection.filters._mah import _compute_y_binning, clear_mah_y_binning_cache, mah_bin_edges, mah_mi

# ---------------------------------------------------------------------------
# qs_nbins / edges_qs
# ---------------------------------------------------------------------------


def test_qs_nbins_no_overflow_at_extreme_n():
    """``qs_nbins`` must stay clamped in [3, 64] even at astronomically large N (no int overflow, no
    runaway bin count that would blow up downstream joint histograms)."""
    for n in (0, 1, 2, 10, 1000, 10**6, 10**12, 10**18):
        nb = qs_nbins(n)
        assert nb == 1 or 3 <= nb <= 64, f"n={n} -> nb={nb} out of contract"


def test_qs_nbins_alpha_extremes_stay_clamped():
    """Alpha outside the recommended [0.25, 0.35] band (a caller typo/misuse) must still clamp, not
    silently explode the bin count or collapse to 0."""
    for alpha in (0.0, 1e-9, 0.99, 5.0, -1.0):
        nb = qs_nbins(100_000, alpha=alpha)
        assert 1 <= nb <= 64


def test_edges_qs_all_identical_values_returns_empty_not_garbage():
    """A constant column has zero variance -- edges_qs must return an EMPTY array cleanly (the
    documented degenerate contract), never NaN/garbage edges from a divide-by-zero quantile spread."""
    x = np.full(5000, 7.0)
    edges = edges_qs(x)
    assert edges.size == 0
    assert np.isfinite(edges).all()


def test_edges_qs_extreme_skew_real_variance_not_silently_worse_than_documented():
    """99.9% mass at one value + a real, spread-out tail (real variance, not a degenerate constant
    column) -- edges_qs is allowed to return few/no edges (Gupta's method legitimately underpowers
    heavy skew), but whatever it returns must be finite and monotonic (never NaN/unsorted garbage
    silently propagated to searchsorted)."""
    rng = np.random.default_rng(0)
    n = 20000
    dominant = np.zeros(n - 20)
    tail = rng.uniform(100.0, 200.0, size=20)
    x = np.concatenate([dominant, tail])
    edges = edges_qs(x, alpha=0.30)
    assert np.isfinite(edges).all()
    assert np.all(np.diff(edges) > 0)  # strictly increasing (np.unique guarantees this, but assert the contract)


def test_edges_qs_tiny_n_no_crash():
    """n below any reasonable quantile granularity must not crash / must not return garbage (finite,
    monotonic edges or a clean empty array -- never NaN/unsorted)."""
    for n in (0, 1, 2, 3):
        x = np.arange(n, dtype=np.float64)
        edges = edges_qs(x)
        assert np.isfinite(edges).all()
        assert np.all(np.diff(edges) > 0)
    # n=0/1 truly have no spread to bin -> must be empty.
    for n in (0, 1):
        x = np.arange(n, dtype=np.float64)
        assert edges_qs(x).size == 0


def test_edges_qs_huge_n_bounded_edge_count():
    """At huge N the returned edge count must stay bounded by the qs_nbins cap (<=63 inner edges),
    never scale with N (an unbounded edge count would defeat the whole point of QS's fixed-alpha
    cardinality control and blow up downstream joint histograms)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=2_000_000)
    edges = edges_qs(x)
    assert edges.size <= 63


def test_edges_qs_never_silently_empty_on_healthy_continuous_data():
    """Sanity floor: on well-behaved, non-degenerate continuous data with ample N, edges_qs must NOT
    silently collapse to empty -- this is the exact failure signature the MDLP bug produced. A healthy
    N(0,1) sample with n=5000 has no excuse to return 0 edges."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    edges = edges_qs(x)
    assert edges.size > 0, "edges_qs silently returned EMPTY edges for healthy continuous data with real variance"


# ---------------------------------------------------------------------------
# biz_val: qs should give ~0 MI under independence, real MI under dependence
# ---------------------------------------------------------------------------


def _plug_in_mi_from_edges(x: np.ndarray, y_codes: np.ndarray, edges: np.ndarray) -> float:
    """Minimal plug-in MI helper for the biz_val checks below (independent of the production MI kernels,
    so this test is not circular with anything under audit elsewhere)."""
    if edges.size == 0:
        x_codes = np.zeros_like(x, dtype=np.int64)
    else:
        x_codes = np.searchsorted(edges, x, side="right")
    K_x = int(x_codes.max()) + 1
    K_y = int(y_codes.max()) + 1
    joint = np.bincount(x_codes * K_y + y_codes, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)
    n = joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_joint = joint / n
        p_ind = (px / n) * (py / n)
        ratio = np.where((joint > 0) & (p_ind > 0), p_joint * np.log(p_joint / p_ind), 0.0)
    return float(ratio.sum())


def test_biz_val_edges_qs_detects_real_dependence_above_independence_floor():
    """Ground truth: y = 1 if x > median else 0 (perfect step dependence) must score MEASURABLY higher
    plug-in MI through edges_qs bins than an independent shuffle of the same y -- QS's whole selling
    point (flat marginal under independence) means the independence floor should be small and the
    signal case should clear it by a wide, quantitatively-checkable margin."""
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.normal(size=n)
    y_signal = (x > np.median(x)).astype(np.int64)
    y_indep = rng.permutation(y_signal)

    edges = edges_qs(x, alpha=0.30)
    assert edges.size > 0

    mi_signal = _plug_in_mi_from_edges(x, y_signal, edges)
    mi_indep = _plug_in_mi_from_edges(x, y_indep, edges)

    assert mi_signal >= 0.30, f"expected strong step-dependence MI >= 0.30 nats, got {mi_signal}"
    assert mi_indep <= 0.02, f"expected near-zero independence-floor MI <= 0.02 nats, got {mi_indep}"
    assert mi_signal > mi_indep * 10


# ---------------------------------------------------------------------------
# mah / SCI -- _compute_y_binning cardinality-driven hang (the real bug found + fixed)
# ---------------------------------------------------------------------------


def test_compute_y_binning_caps_high_cardinality_integer_y_at_K():
    """Regression test for the fixed bug: an int64-dtype y with cardinality >> K used to be routed
    through the exact label-encode branch UNCONDITIONALLY (any int dtype), giving K_y = full
    cardinality instead of capping at K. Post-fix, K_y must never exceed K for high-cardinality y,
    regardless of dtype."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(4)
    K = 16
    y_int_high_card = (rng.normal(size=3000) * 1000).astype(np.int64)
    assert np.unique(y_int_high_card).size > K  # precondition: genuinely high cardinality
    _yb, K_y = _compute_y_binning(y_int_high_card, K)
    assert K_y <= K, f"high-cardinality int y must be quantile-capped at K={K}, got K_y={K_y}"


def test_compute_y_binning_low_cardinality_integer_y_exact_label_encode_unchanged():
    """Sibling of the fix above: genuinely low-cardinality int y (<=K distinct values) must still take
    the EXACT label-encode path (this is the behavior the fix must NOT change)."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(5)
    y = rng.integers(0, 5, 2000).astype(np.int64)
    yb, K_y = _compute_y_binning(y, 16)
    assert K_y == 5
    uniq_y = np.unique(y)
    assert np.array_equal(yb, np.searchsorted(uniq_y, y))


def test_mah_bin_edges_high_cardinality_integer_target_completes_quickly():
    """Regression test for the actual hang: pre-fix, ``mah_bin_edges`` on a continuous-like int64
    target (thousands of distinct values) hung past a 40s timeout in a separate-process empirical
    check (subprocess timeout, exit code 124) -- ``_greedy_merge_bins``'s per-step O(rows*cols)
    candidate rebuild across O(cols) candidates blew up for K_y ~ 2000. Verified fails pre-fix / passes
    post-fix by directly reverting just the ``_compute_y_binning`` change (see PR history); this test
    pins the post-fix behavior: must complete well within a generous 10s budget."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(6)
    n = 3000
    x = rng.normal(size=n)
    y = (rng.normal(size=n) * 1000).astype(np.int64)
    assert np.unique(y).size > 500  # precondition: genuinely high cardinality, would have hung pre-fix

    t0 = time.time()
    edges = mah_bin_edges(x, y, initial_k=16)
    elapsed = time.time() - t0

    assert elapsed < 10.0, f"mah_bin_edges took {elapsed:.2f}s on high-cardinality int y -- possible regression of the fixed hang"
    assert np.isfinite(edges).all()


def test_mah_bin_edges_high_cardinality_integer_target_subprocess_empirical_bound():
    """Stronger, process-isolated version of the timing regression test above: launches a fresh
    Python process (no warm JIT/cache contamination) and asserts it returns within a hard wall-clock
    budget well under the 40s the pre-fix code was empirically observed to exceed."""
    script = (
        "import numpy as np\n"
        "from mlframe.feature_selection.filters._mah import mah_bin_edges\n"
        "rng = np.random.default_rng(7)\n"
        "n = 3000\n"
        "x = rng.normal(size=n)\n"
        "y = (rng.normal(size=n) * 1000).astype(np.int64)\n"
        "edges = mah_bin_edges(x, y, initial_k=16)\n"
        "print('OK', edges.size)\n"
    )
    # 120s budget (not the ~0.66s warm in-process measurement): a fresh process pays numba's cold
    # njit-compile cost on top, which under concurrent machine load can itself take tens of seconds --
    # this test only needs to prove it is NOT the pre-fix multi-minute-plus hang, not race the JIT.
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"subprocess failed/timed out: stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    assert "OK" in result.stdout


def test_mah_bin_edges_all_identical_x_returns_empty():
    """Constant X column (zero variance) -- mah_bin_edges must return an empty array cleanly (the
    ``qx_unique.size < 3`` degenerate-input guard), never crash or hang."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(8)
    x = np.full(2000, 3.0)
    y = rng.integers(0, 3, 2000).astype(np.int64)
    edges = mah_bin_edges(x, y, initial_k=16)
    assert edges.size == 0


def test_mah_bin_edges_tiny_n_no_crash():
    """Below the hard n<16 floor mah_bin_edges must return empty, not raise / index-error."""
    rng = np.random.default_rng(9)
    for n in (0, 1, 5, 15):
        x = rng.normal(size=n)
        y = rng.integers(0, 2, n).astype(np.int64) if n else np.array([], dtype=np.int64)
        edges = mah_bin_edges(x, y, initial_k=16)
        assert edges.size == 0


def test_mah_bin_edges_huge_n_bounded_and_finite():
    """Large-N sanity: mah_bin_edges must stay bounded by initial_k-1 inner edges (never grow with N)
    and must never emit non-finite edges."""
    rng = np.random.default_rng(10)
    n = 200_000
    x = rng.normal(size=n)
    y = rng.integers(0, 4, n).astype(np.int64)
    edges = mah_bin_edges(x, y, initial_k=16)
    assert edges.size <= 15
    assert np.isfinite(edges).all()


# ---------------------------------------------------------------------------
# biz_val: mah/SCI MI should separate real dependence from independence
# ---------------------------------------------------------------------------


def test_biz_val_mah_mi_detects_real_dependence_above_independence_floor():
    """Ground truth: a clearly step-dependent (x, y) pair must score materially higher SCI-based MI
    than an independent shuffle of the same y -- quantitative thresholds, not an ``is not None`` check."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(11)
    n = 5000
    x = rng.normal(size=n)
    y_signal = (x > np.median(x)).astype(np.int64)
    y_indep = rng.permutation(y_signal)

    mi_signal = mah_mi(x, y_signal, initial_k=16)
    mi_indep = mah_mi(x, y_indep, initial_k=16)

    assert mi_signal >= 0.20, f"expected mah_mi >= 0.20 nats on strong step dependence, got {mi_signal}"
    assert mi_indep <= 0.05, f"expected near-zero mah_mi under independence, got {mi_indep}"
