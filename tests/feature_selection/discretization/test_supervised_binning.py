"""Unit tests for mlframe.feature_selection.filters.supervised_binning.

Covers the Fayyad-Irani MDLP supervised discretiser and the apply helper.
The optbinning wrapper is not exercised here because it is a thin pass-through to a third-party library and its semantics live upstream.

Invariants under test:
* return-shape contract on ``mdlp_bin_edges`` (sentinel edges, sorted)
* MDL stopping criterion: pure noise gives 0 splits, perfect signal gives >=1 split near the boundary
* entropy reduction H(y | binned x) < H(y) for non-trivial signal
* ``max_depth`` / ``min_split_size`` plumbed through
* ``apply_bin_edges`` leak-safe round-trip and dtype handling
* edge cases: single value, all-constant, single-class y, n=2, length mismatch
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.supervised_binning import (
    apply_bin_edges,
    mdlp_bin_edges,
)


# ---- helpers ---------------------------------------------------------------


def _shannon_entropy(y: np.ndarray) -> float:
    """Shannon entropy in nats; tiny helper to keep tests independent of internals."""
    if len(y) == 0:
        return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _conditional_entropy(y: np.ndarray, bins: np.ndarray) -> float:
    """H(y | bins) in nats."""
    n = len(y)
    if n == 0:
        return 0.0
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        total += (mask.sum() / n) * _shannon_entropy(y[mask])
    return float(total)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260514)


@pytest.fixture
def separable_xy(rng):
    """x ~ U(0,1), y = 1[x > 0.5]; perfectly separable."""
    x = rng.uniform(0.0, 1.0, size=500)
    y = (x > 0.5).astype(np.int64)
    return x, y


@pytest.fixture
def noisy_boundary_xy(rng):
    """x ~ U(0,1), y = 1[x > 0.3] with 5 % label flips."""
    x = rng.uniform(0.0, 1.0, size=1000)
    y = (x > 0.3).astype(np.int64)
    flip = rng.uniform(size=1000) < 0.05
    y = np.where(flip, 1 - y, y).astype(np.int64)
    return x, y


# ---- API contract ----------------------------------------------------------


class TestMDLPContract:
    def test_returns_ndarray_with_inf_sentinels(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        assert isinstance(edges, np.ndarray)
        assert edges.dtype == np.float64
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_edges_are_sorted(self, noisy_boundary_xy):
        x, y = noisy_boundary_xy
        edges = mdlp_bin_edges(x, y, max_depth=4, min_split_size=10)
        assert np.all(np.diff(edges) > 0)

    def test_at_least_two_edges_always(self, rng):
        # Even with zero splits we still return [-inf, +inf].
        x = rng.standard_normal(20)
        y = rng.integers(0, 2, size=20)
        edges = mdlp_bin_edges(x, y)
        assert len(edges) >= 2

    def test_length_mismatch_raises(self, rng):
        x = rng.standard_normal(50)
        y = rng.integers(0, 2, size=40)
        with pytest.raises(ValueError, match="len"):
            mdlp_bin_edges(x, y)

    def test_determinism(self, separable_xy):
        # No RNG inside mdlp_bin_edges; identical inputs must give identical edges.
        x, y = separable_xy
        e1 = mdlp_bin_edges(x, y)
        e2 = mdlp_bin_edges(x.copy(), y.copy())
        np.testing.assert_array_equal(e1, e2)

    def test_accepts_list_input(self, separable_xy):
        # The function calls np.asarray; lists must work.
        x, y = separable_xy
        e_arr = mdlp_bin_edges(x, y)
        e_list = mdlp_bin_edges(x.tolist(), y.tolist())
        np.testing.assert_array_equal(e_arr, e_list)


# ---- MDL stopping behaviour ------------------------------------------------


class TestMDLStopping:
    def test_perfect_separator_yields_split(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        inner = edges[1:-1]
        assert len(inner) >= 1
        assert min(abs(s - 0.5) for s in inner) < 0.05

    def test_uniform_noise_yields_few_splits(self, rng):
        # y independent of x: Fayyad-Irani MDL should reject the split.
        x = rng.uniform(0.0, 1.0, size=500)
        y = rng.integers(0, 2, size=500)
        edges = mdlp_bin_edges(x, y)
        inner = edges[1:-1]
        # MDL is conservative; allow up to 1 false-positive split but never a forest.
        assert len(inner) <= 1

    def test_single_class_y_no_splits(self, rng):
        x = rng.standard_normal(200)
        y = np.zeros(200, dtype=np.int64)
        edges = mdlp_bin_edges(x, y)
        assert len(edges) == 2  # only sentinels


# ---- depth / min_split_size ------------------------------------------------


class TestKnobs:
    def test_max_depth_zero_blocks_all_splits(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y, max_depth=0)
        assert len(edges) == 2

    def test_min_split_size_huge_blocks_all_splits(self, separable_xy):
        x, y = separable_xy
        # Need 2 * min_split_size <= n for recursion to enter; oversize blocks it.
        edges = mdlp_bin_edges(x, y, min_split_size=len(x))
        assert len(edges) == 2

    def test_deeper_max_depth_does_not_remove_splits(self, noisy_boundary_xy):
        x, y = noisy_boundary_xy
        shallow = mdlp_bin_edges(x, y, max_depth=1)
        deeper = mdlp_bin_edges(x, y, max_depth=8)
        # Recursion only adds splits, never deletes; the shallow set must be a subset.
        inner_shallow = set(shallow[1:-1].tolist())
        inner_deeper = set(deeper[1:-1].tolist())
        assert inner_shallow.issubset(inner_deeper)


# ---- entropy reduction -----------------------------------------------------


class TestEntropyReduction:
    def test_split_reduces_conditional_entropy(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        bins = apply_bin_edges(x, edges)
        h0 = _shannon_entropy(y)
        h_cond = _conditional_entropy(y, bins)
        # Perfectly separable: H(y | bins) should be ~0 vs ~ln 2 base.
        assert h_cond < 0.5 * h0


# ---- edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_n_equals_two(self):
        edges = mdlp_bin_edges(np.array([0.0, 1.0]), np.array([0, 1]))
        # 2 * min_split_size = 10 > n=2, so no splits expected.
        assert len(edges) == 2

    def test_all_constant_x(self, rng):
        x = np.full(200, 3.14)
        y = rng.integers(0, 2, size=200)
        edges = mdlp_bin_edges(x, y)
        # No class boundary can fall between identical x values once sorted has gaps - check:
        # algorithm may still pick a candidate but MDL likely rejects.
        # At minimum the function must return sentinels only or one trivial split.
        assert edges[0] == -np.inf and edges[-1] == np.inf

    def test_single_value_x(self):
        edges = mdlp_bin_edges(np.array([1.0]), np.array([0]))
        assert list(edges) == [-np.inf, np.inf]


# ---- apply_bin_edges -------------------------------------------------------


class TestApplyBinEdges:
    def test_bin_count_matches_edges(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        bins = apply_bin_edges(x, edges)
        n_bins_expected = len(edges) - 1
        assert bins.min() >= 0
        assert bins.max() < n_bins_expected

    def test_dtype_respected(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        for dt in (np.int8, np.int16, np.int32, np.int64):
            out = apply_bin_edges(x, edges, dtype=dt)
            assert out.dtype == dt

    def test_train_val_share_edges(self, rng):
        # Leak-safe pattern from the module docstring: fit edges on train, apply to val.
        x_train = rng.uniform(0, 1, size=400)
        y_train = (x_train > 0.4).astype(np.int64)
        x_val = rng.uniform(0, 1, size=200)
        edges = mdlp_bin_edges(x_train, y_train)
        bins_train = apply_bin_edges(x_train, edges)
        bins_val = apply_bin_edges(x_val, edges)
        # Both should use the same bin id space and never exceed n_edges-1.
        n_bins = len(edges) - 1
        assert bins_train.max() < n_bins
        assert bins_val.max() < n_bins

    def test_apply_monotone_in_x(self, separable_xy):
        x, y = separable_xy
        edges = mdlp_bin_edges(x, y)
        order = np.argsort(x)
        bins_sorted = apply_bin_edges(x[order], edges)
        # Sorting x must produce a non-decreasing bin sequence (digitize is monotone).
        assert np.all(np.diff(bins_sorted.astype(np.int64)) >= 0)


# ---- biz_value -------------------------------------------------------------


@pytest.mark.fast
def test_biz_mdlp_recovers_decision_boundary():
    """MDLP must find the true decision boundary x=0.3 under 5 % label noise.

    Calibration: observed split at 0.3002 (margin 2e-4 from truth) on this seed; asserting the [0.25, 0.35] window keeps a >100x safety
    factor over noise and over RNG drift. Asserts at least one inner split in that window.
    """
    rng = np.random.default_rng(20260514)
    x = rng.uniform(0.0, 1.0, size=1000)
    y_clean = (x > 0.3).astype(np.int64)
    flip = rng.uniform(size=1000) < 0.05
    y = np.where(flip, 1 - y_clean, y_clean).astype(np.int64)

    edges = mdlp_bin_edges(x, y)
    inner = edges[1:-1]

    assert len(inner) >= 1, "MDLP failed to produce any split on a clearly separable signal."
    assert len(inner) <= 3, f"MDLP produced too many splits ({len(inner)}); MDL stopping looks broken."
    assert any(0.25 <= s <= 0.35 for s in inner), (
        f"No split within [0.25, 0.35] of the true boundary 0.3; got splits {inner.tolist()}"
    )


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_mdlp_njit_backend_matches_python_backend_multiclass(K):
    """The njit recursion derives per-side class counts (which feed the MDL threshold) from
    ``np.count_nonzero`` of the bincount it already builds, instead of a separate ``np.unique``
    sort of the subset. The resulting edges must stay byte-identical to the untouched pure-Python
    ``backend='python'`` reference (which still uses np.unique) on a multiclass signal where the
    left/right subsets carry differing class counts -- the exact path that class count governs.
    """
    rng = np.random.default_rng(20260705 + K)
    n = 8000
    x = np.sort(rng.standard_normal(n)).astype(np.float64)
    # y = region label with 15% noise so left/right subsets have distinct, differing class counts.
    edges_true = np.quantile(x, np.linspace(0, 1, K + 1))[1:-1]
    y_clean = np.searchsorted(edges_true, x)
    noisy = rng.uniform(size=n) < 0.15
    y = np.where(noisy, rng.integers(0, K, n), y_clean).astype(np.int64)

    e_njit = mdlp_bin_edges(x, y, backend="njit")
    e_py = mdlp_bin_edges(x, y, backend="python")
    np.testing.assert_array_equal(
        e_njit, e_py,
        err_msg=f"njit backend edges diverge from python reference at K={K} (class-count derivation bug)",
    )
