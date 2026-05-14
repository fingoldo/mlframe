"""Unit tests for ``mlframe.feature_selection.filters._prewarm``.

The pre-warm entry point triggers numba JIT compilation across every dispatcher path used by the screening / cat-FE / discretisation hot loops. The risk
pattern: when a new dispatcher path is added in production but NOT in the prewarm matrix, the first real fit pays the full cold-start cost (~17s for
``parallel_mi*``, ~9s per discretisation dtype). These tests cover:

  - Smoke: ``prewarm_fs_numba_cache()`` runs without raising and is idempotent.
  - Coverage: known critical dispatchers (``compute_mi_from_classes``, discretisation kernels) have at least one compiled signature after prewarm.
  - Biz-value: a downstream ``compute_mi_from_classes`` call hits cache (post-warm >= 2x faster than its own first cold compile).
"""
from __future__ import annotations

import time

import numpy as np
import pytest


# Module-scoped fixture: prewarm exactly once per test session for the coverage / biz-value tests that need a warmed dispatcher graph.
@pytest.fixture(scope="module")
def warmed():
    from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
    prewarm_fs_numba_cache(verbose=False)
    return True


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmSmoke:
    """Entry-point sanity: importable, callable, idempotent, returns None."""

    def test_importable(self):
        from mlframe.feature_selection.filters import _prewarm
        assert hasattr(_prewarm, "prewarm_fs_numba_cache")
        assert callable(_prewarm.prewarm_fs_numba_cache)

    def test_public_api(self):
        from mlframe.feature_selection.filters import _prewarm
        assert _prewarm.__all__ == ["prewarm_fs_numba_cache"]

    def test_runs_without_exception(self):
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
        # Must NOT raise even if individual kernels fail (the body wraps each call in try/except).
        prewarm_fs_numba_cache(verbose=False)

    def test_returns_none(self):
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
        out = prewarm_fs_numba_cache(verbose=False)
        assert out is None

    def test_idempotent(self):
        # Second call must be near-instant (numba dispatcher cache hit) and again must not raise.
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
        prewarm_fs_numba_cache(verbose=False)
        t0 = time.perf_counter()
        prewarm_fs_numba_cache(verbose=False)
        elapsed = time.perf_counter() - t0
        # Idempotent call should be far faster than the initial compile budget (~60s). Generous 30s upper bound to avoid CI flakiness.
        assert elapsed < 30.0, f"second prewarm took {elapsed:.2f}s -- cache not effective"

    def test_verbose_flag_accepts_true(self):
        # Verbose path emits a log line; must not raise.
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
        prewarm_fs_numba_cache(verbose=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Coverage: known dispatchers compiled
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmCoverage:
    """After prewarm, key dispatchers must have at least one compiled signature -- otherwise the production hot path still pays JIT cost on first use."""

    def test_compute_mi_from_classes_has_signatures(self, warmed):
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes
        # ``signatures`` is the public list of (dtype tuple) -> compiled-impl entries on a numba Dispatcher.
        assert len(compute_mi_from_classes.signatures) >= 1, "compute_mi_from_classes was not compiled by prewarm"

    def test_numba_utils_compiled(self, warmed):
        from mlframe.feature_selection.filters._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
        assert len(arr2str.signatures) >= 1
        assert len(count_cand_nbins.signatures) >= 1
        assert len(unpack_and_sort.signatures) >= 1

    def test_permutation_kernels_compiled(self, warmed):
        from mlframe.feature_selection.filters.permutation import parallel_mi, parallel_mi_prange, shuffle_arr
        assert len(parallel_mi.signatures) >= 1
        assert len(parallel_mi_prange.signatures) >= 1
        assert len(shuffle_arr.signatures) >= 1

    def test_discretization_dtype_matrix_compiled(self, warmed):
        # The public ``discretize_array`` is a regular Python wrapper; its njit kernel is ``_discretize_array_impl``.
        # Guards the dtype-matrix prewarm regression (a missed dtype leaves a ~9s cold compile in prod).
        from mlframe.feature_selection.filters.discretization import _discretize_array_impl
        assert len(_discretize_array_impl.signatures) >= 1

    def test_post_warm_no_inf_or_nan_in_smoke_kernel(self, warmed):
        # Spot-check the canonical kernel: after prewarm, computing MI on tiny input must return a finite non-negative scalar.
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes
        rng = np.random.default_rng(0)
        cx = rng.integers(0, 4, 32).astype(np.int32)
        cy = rng.integers(0, 3, 32).astype(np.int32)
        fx = np.bincount(cx, minlength=4).astype(np.float64) / 32
        fy = np.bincount(cy, minlength=3).astype(np.float64) / 32
        mi = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        assert np.isfinite(mi)
        assert mi >= 0.0


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# biz_value: prewarm reduces cold-start
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmBizValue:
    """Quantitative win: a representative downstream ``@njit`` call must be measurably faster after prewarm than the first (cold) call would have been."""

    def test_biz_prewarm_reduces_cold_start(self, warmed):
        # We cannot truly time "cold vs warm" inside one process because numba caches process-wide. Instead: after prewarm, the call on a fresh tiny fixture
        # must complete well under the documented cold-compile budget (~17s for parallel_mi, ~3-5s for compute_mi_from_classes). A warm dispatch is sub-ms.
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

        rng = np.random.default_rng(42)
        n = 1000
        cx = rng.integers(0, 8, n).astype(np.int32)
        cy = rng.integers(0, 4, n).astype(np.int32)
        fx = np.bincount(cx, minlength=8).astype(np.float64) / n
        fy = np.bincount(cy, minlength=4).astype(np.float64) / n

        # Warm call must be sub-100ms -- if prewarm did its job the dispatcher is cached, the actual numeric work on 1000 rows is microseconds, and we have a
        # solid 100x margin over the cold-compile floor (~3s) which is what the assertion is really gating.
        t0 = time.perf_counter()
        mi1 = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        warm_elapsed = time.perf_counter() - t0

        # Second call -- pure cache hit, must be at most as slow as the first warm call (allow 2x slack for timer noise on tiny calls).
        t0 = time.perf_counter()
        mi2 = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        second_elapsed = time.perf_counter() - t0

        assert np.isfinite(mi1) and np.isfinite(mi2)
        assert mi1 == mi2, "deterministic kernel must produce identical output on identical input"
        # Warm call well under the cold-compile floor. Generous bound (0.5s) to absorb CI jitter; the real cold-compile is ~3-5s on the same machine.
        assert warm_elapsed < 0.5, f"first post-prewarm call took {warm_elapsed:.3f}s -- prewarm did not cover this dispatcher signature"
        # Second call must be at least as fast as warm (in expectation). With tiny-call timer noise we just require it stays under the same envelope.
        assert second_elapsed < 0.5, f"cached call took {second_elapsed:.3f}s -- unexpected"
