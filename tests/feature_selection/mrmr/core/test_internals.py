"""Characterization unit tests for internal MRMR helpers (etap 0b).

Established BEFORE the move-and-fix etapes (2-9) so subtle regressions in
helper functions are caught even when integration ``MRMR.fit`` smoke tests
pass. Many helpers have ZERO direct coverage in the existing 43-test suite
(only ``entropy``, ``compute_mi_from_classes``, ``discretize_array``,
``categorize_dataset`` were directly tested before this file).

The tests intentionally import from the public surface (``mlframe.feature_
selection.filters``) so they keep working through the move etapes via the
``__init__.py`` re-exports. Tests added here for functions that DON'T appear
in ``__init__.py`` use ``pytest.importorskip`` patterns where appropriate.
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import (
    MRMR,
    arr2str,
    categorize_dataset,
    compute_mi_from_classes,
    discretize_array,
    distribute_permutations,
    entropy,
    merge_vars,
    mi_direct,
    parallel_mi,
)
from mlframe.feature_selection.filters.permutation import parallel_mi_prange


# =============================================================================
# arr2str (B12 silent correctness regression)
# =============================================================================


class TestArr2StrCollisions:
    """The pre-B12 ``arr2str`` collapses multiset variants to identical strings.

    Routine collisions for ``n_features >= 10 AND interactions_max_order >= 2``
    (see ``_benchmarks/_results/collision_census_pre_refactor.json``). After
    B12 with the ``_`` separator + preallocated buffer impl the test below
    must xfail-flip into pass on stage 7.
    """

    def test_distinct_multisets_yield_distinct_keys(self):
        a = arr2str(np.array([1, 11], dtype=np.int64))
        b = arr2str(np.array([1, 1, 1], dtype=np.int64))
        assert a != b, f"collision: arr2str([1,11])={a!r} == arr2str([1,1,1])={b!r}"

    def test_arr2str_deterministic(self):
        a = arr2str(np.array([3, 1, 2], dtype=np.int64))
        b = arr2str(np.array([3, 1, 2], dtype=np.int64))
        assert a == b


# =============================================================================
# mi_direct (CPU permutation test wrapper)
# =============================================================================


class TestMiDirect:
    """Smoke + the B22 zero-permutation guard."""

    @pytest.fixture
    def factor_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 1000
        # Dependent x->y; mi should be > 0.
        a = rng.integers(0, 5, size=n).astype(np.int32)
        b = (a + rng.integers(0, 2, size=n)) % 5
        b = b.astype(np.int32)
        # factors_data layout: (n_samples, n_features) per categorize_dataset.
        factors = np.column_stack([a, b]).astype(np.int32)
        nbins = np.array([5, 5], dtype=np.int32)
        return a, b, factors, nbins

    def test_mi_direct_dependent_returns_positive(self, factor_data):
        a, b, factors, nbins = factor_data
        original_mi, conf = mi_direct(
            factors_data=factors,
            x=(0,),
            y=(1,),
            factors_nbins=nbins,
            npermutations=10,
            n_workers=1,
        )
        assert original_mi > 0
        assert 0 <= conf <= 1

    def test_mi_direct_independent_returns_low_or_zero(self):
        rng = np.random.default_rng(123)
        n = 1000
        a = rng.integers(0, 5, size=n).astype(np.int32)
        b = rng.integers(0, 5, size=n).astype(np.int32)
        factors = np.column_stack([a, b]).astype(np.int32)
        nbins = np.array([5, 5], dtype=np.int32)
        original_mi, _ = mi_direct(
            factors_data=factors,
            x=(0,),
            y=(1,),
            factors_nbins=nbins,
            npermutations=20,
            n_workers=1,
            min_nonzero_confidence=0.99,
        )
        assert original_mi >= 0
        assert original_mi < 0.1

    def test_mi_direct_npermutations_zero_returns_confidence_zero(self, factor_data):
        """Calls with ``npermutations=0`` must not crash. After B22 the caller
        path also returns confidence=0 cleanly."""
        a, b, factors, nbins = factor_data
        original_mi, conf = mi_direct(
            factors_data=factors,
            x=(0,),
            y=(1,),
            factors_nbins=nbins,
            npermutations=0,
            n_workers=1,
        )
        assert original_mi >= 0
        assert conf == 0


# =============================================================================
# parallel_mi (njit worker for mi_direct's joblib pool)
# =============================================================================


class TestPhase1PrangeReproducibility:
    """Phase 1 (post-plan etap 14): parallel_mi_prange must produce
    identical (nfailed, nchecked) for the same (base_seed, npermutations)
    across any n_workers (or implementation that wraps it). This invariant
    is the formal acceptance gate for the prange path."""

    @pytest.fixture
    def perm_data(self):
        rng = np.random.default_rng(0)
        n = 2000
        cx = rng.integers(0, 4, size=n).astype(np.int32)
        cy = rng.integers(0, 4, size=n).astype(np.int32)
        fx = np.bincount(cx, minlength=4).astype(np.float64) / n
        fy = np.bincount(cy, minlength=4).astype(np.float64) / n
        return cx, fx, cy, fy

    def test_parallel_mi_prange_deterministic(self, perm_data):
        cx, fx, cy, fy = perm_data
        a = parallel_mi_prange(cx, fx, cy, fy, npermutations=50, original_mi=0.5, base_seed=np.uint64(42))
        b = parallel_mi_prange(cx, fx, cy, fy, npermutations=50, original_mi=0.5, base_seed=np.uint64(42))
        assert a == b, f"non-deterministic: {a} vs {b}"

    def test_parallel_mi_prange_different_seeds_diverge(self, perm_data):
        cx, fx, cy, fy = perm_data
        # original_mi=0.001 -- small enough that random permutations can
        # exceed it on a shuffle that happens to be slightly correlated.
        # With seeds=1 and seeds=2 the per-iter LCG produces different
        # streams so nfailed counts must diverge across many trials.
        a = parallel_mi_prange(cx, fx, cy, fy, npermutations=200, original_mi=0.001, base_seed=np.uint64(1))
        b = parallel_mi_prange(cx, fx, cy, fy, npermutations=200, original_mi=0.001, base_seed=np.uint64(2))
        assert a != b, f"expected divergence on different seeds: {a} vs {b}"

    def test_parallel_mi_prange_zero_permutations(self, perm_data):
        cx, fx, cy, fy = perm_data
        nfailed, nchecked = parallel_mi_prange(cx, fx, cy, fy, npermutations=0, original_mi=0.5, base_seed=np.uint64(1))
        assert nfailed == 0 and nchecked == 0


class TestParallelMi:
    def test_parallel_mi_zero_permutations_no_crash(self):
        rng = np.random.default_rng(7)
        n = 100
        cx = rng.integers(0, 3, size=n).astype(np.int32)
        cy = rng.integers(0, 3, size=n).astype(np.int32)
        fx = np.bincount(cx, minlength=3).astype(np.float64) / n
        fy = np.bincount(cy, minlength=3).astype(np.float64) / n
        nfailed, nchecked = parallel_mi(
            classes_x=cx,
            freqs_x=fx,
            classes_y=cy,
            freqs_y=fy,
            npermutations=0,
            original_mi=0.5,
            max_failed=1,
        )
        assert nfailed == 0
        assert nchecked == 0


# =============================================================================
# distribute_permutations
# =============================================================================


class TestDistributePermutations:
    @pytest.mark.parametrize(
        "n,workers",
        [(10, 4), (100, 4), (1, 1), (5, 5), (100, 1)],
    )
    def test_workload_sums_to_n(self, n, workers):
        wl = distribute_permutations(n, workers)
        assert sum(wl) == n
        assert len(wl) == workers

    def test_workload_balance_extreme_remainder(self):
        # 11 permutations across 4 workers: 2,2,2,5 (last absorbs extra) or similar.
        wl = distribute_permutations(11, 4)
        assert sum(wl) == 11
        assert len(wl) == 4
        # All chunks non-negative.
        assert all(w >= 0 for w in wl)


# =============================================================================
# merge_vars
# =============================================================================


class TestMergeVars:
    def test_single_var_round_trips(self):
        rng = np.random.default_rng(0)
        n = 500
        a = rng.integers(0, 4, size=n).astype(np.int32)
        factors = a.reshape(-1, 1)  # (n_samples, 1 feature)
        nbins = np.array([4], dtype=np.int32)
        classes, freqs, _ = merge_vars(
            factors_data=factors,
            vars_indices=(0,),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        assert classes.shape == (n,)
        assert np.isclose(freqs.sum(), 1.0, atol=1e-6)

    def test_two_var_merge_yields_higher_cardinality(self):
        rng = np.random.default_rng(1)
        n = 1000
        a = rng.integers(0, 3, size=n).astype(np.int32)
        b = rng.integers(0, 4, size=n).astype(np.int32)
        factors = np.column_stack([a, b]).astype(np.int32)
        nbins = np.array([3, 4], dtype=np.int32)
        classes_a, _, _ = merge_vars(
            factors_data=factors,
            vars_indices=(0,),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        classes_ab, _, _ = merge_vars(
            factors_data=factors,
            vars_indices=(0, 1),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        assert len(np.unique(classes_ab)) >= len(np.unique(classes_a))


# =============================================================================
# compute_mi_from_classes
# =============================================================================


class TestComputeMiFromClasses:
    def test_mi_self_equals_entropy(self):
        rng = np.random.default_rng(5)
        n = 1000
        a = rng.integers(0, 5, size=n).astype(np.int32)
        fa = np.bincount(a, minlength=5).astype(np.float64) / n
        mi_self = compute_mi_from_classes(a, fa, a, fa)
        h = entropy(fa)
        assert np.isclose(mi_self, h, rtol=1e-9)

    def test_mi_independent_near_zero(self):
        rng = np.random.default_rng(10)
        n = 5000
        a = rng.integers(0, 4, size=n).astype(np.int32)
        b = rng.integers(0, 4, size=n).astype(np.int32)
        fa = np.bincount(a, minlength=4).astype(np.float64) / n
        fb = np.bincount(b, minlength=4).astype(np.float64) / n
        mi = compute_mi_from_classes(a, fa, b, fb)
        assert mi < 0.05


# =============================================================================
# discretize_array (B10 NaN handling fixture, present-state characterization)
# =============================================================================


class TestDiscretizeArray:
    def test_uniform_method_smoke(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000).astype(np.float64)
        out = discretize_array(arr=x, n_bins=10, method="uniform", dtype=np.int32)
        assert out.shape == (2000,)
        assert out.dtype == np.int32
        assert out.min() >= 0
        assert out.max() < 10

    def test_quantile_method_smoke(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000).astype(np.float64)
        out = discretize_array(arr=x, n_bins=10, method="quantile", dtype=np.int32)
        assert out.shape == (2000,)
        # Quantile binning should produce a roughly uniform distribution.
        counts = np.bincount(out)
        assert counts.max() / counts.min() < 3.0

    def test_nan_handling_does_not_raise(self):
        """sklearn >= 1.3 KBinsDiscretizer silently routes NaN to its own bin.
        B10 task is to make pandas/polars paths agree on the strategy explicitly
        (current pandas path silently fillna(0.0), polars path propagates NaN)."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=200).astype(np.float64)
        x[::20] = np.nan
        out = discretize_array(arr=x, n_bins=5, method="uniform", dtype=np.int32)
        assert out.shape == (200,)


# =============================================================================
# categorize_dataset (smoke)
# =============================================================================


class TestCategorizeDataset:
    def test_pandas_numeric_smoke(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"f_{i}": rng.normal(size=500) for i in range(5)})
        cat, cols, nbins = categorize_dataset(df=df, n_bins=8, method="quantile")
        # categorize_dataset returns (n_samples, n_features) layout (sklearn standard).
        assert cat.shape == (500, 5)
        assert len(cols) == 5
        assert all(b == 8 for b in nbins)


# =============================================================================
# MRMR pickle (B27 __setstate__ shim placeholder + plain round-trip)
# =============================================================================


class TestMrmrPickle:
    def test_unfitted_pickle_round_trip(self):
        m = MRMR(quantization_nbins=8, n_jobs=1, verbose=0)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert m2.quantization_nbins == 8

    def test_fitted_pickle_round_trip_smoke(self):
        rng = np.random.default_rng(0)
        n, p = 200, 10
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f_{i}" for i in range(p)])
        y = (X["f_0"] + X["f_1"] > 0).astype(int).to_numpy()
        m = MRMR(
            quantization_nbins=5,
            full_npermutations=2,
            baseline_npermutations=1,
            n_jobs=1,
            verbose=0,
            cv=2,
        )
        m.fit(X, y)
        support_before = m.support_.copy()
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        np.testing.assert_array_equal(m2.support_, support_before)


# =============================================================================
# entropy edge case
# =============================================================================


class TestEntropyEdges:
    def test_single_bin_entropy_zero(self):
        freqs = np.array([1.0])
        assert entropy(freqs) == 0.0

    def test_two_bin_uniform_entropy_ln2(self):
        freqs = np.array([0.5, 0.5])
        assert math.isclose(entropy(freqs), math.log(2), rel_tol=1e-9)
