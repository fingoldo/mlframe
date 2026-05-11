"""Numba-JIT pre-warm helpers for the feature_selection.filters
package.

Calling ``prewarm_fs_numba_cache()`` once at process start triggers
the JIT compile of all hot ``@njit`` kernels in:

  - ``cat_interactions``  (cat-FE permutation tests, pair search)
  - ``discretization``    (quantile / uniform binning)
  - ``info_theory``       (mutual-information primitives)

Without prewarm the first ``MRMR.fit`` call pays ~60 s of cumulative
JIT cold-start (verified on the 1M-row regression+MRMR c0089 profile:
JIT-related cumtime sat at 60.8 s before this hook, all attributed
through ``run_cat_interaction_step`` and other FS callsites).
``@njit(cache=True)`` writes the compiled object to disk only after
a SUCCESSFUL first run, so cold-start latency hits every process
that hasn't yet cached. Prewarm overlaps it with the suite's startup
phase rather than the timed fit phase.

This module is intentionally lazy: it imports the kernels only when
``prewarm_fs_numba_cache()`` is called, so importing
``mlframe.feature_selection.filters`` is fast even when the user
doesn't need MRMR.

Idempotent: numba caches compilations process-wide, so calling
multiple times is essentially free after the first.
"""

from __future__ import annotations

import numpy as np


def prewarm_fs_numba_cache(verbose: bool = False) -> None:
    """Trigger JIT compilation of feature_selection numba kernels.

    Safe to call from anywhere; no-op on numba-unavailable systems.
    Each kernel is exercised on a tiny (n=10) synthetic input that
    matches its production signature -- enough for numba to compile
    + cache the binary, not enough to measure or affect.
    """
    import time
    import logging
    _log = logging.getLogger(__name__)
    _t0 = time.perf_counter()
    try:
        from .cat_interactions import (
            _marginal_screen_njit,
            _pair_search_kernel_njit,
            _pair_search_kernel_weighted_njit,
            _count_nfailed_joint_indep_prange,
            _shuffle_and_compute_three_mis,
            _conditional_shuffle_within_strata,
            _full_conditional_shuffle_ipf,
            _group_aware_shuffle,
        )
        from .info_theory import compute_mi_from_classes
        from .discretization import discretize_2d_array, discretize_array
    except ImportError:
        return
    except Exception:
        # numba unavailable / module misconfigured -- not fatal.
        return

    n = 32
    K_x = 4
    K_y = 3
    rng = np.random.default_rng(0)

    # Cat-interaction permutation kernels (the joint-MI hot path).
    classes_pair = rng.integers(0, K_x * 2, n).astype(np.int32)
    classes_x1 = rng.integers(0, K_x, n).astype(np.int32)
    classes_x2 = rng.integers(0, K_x, n).astype(np.int32)
    classes_y = rng.integers(0, K_y, n).astype(np.int32)
    classes_y_safe = classes_y.copy()
    freqs_pair = np.bincount(classes_pair, minlength=K_x * 2).astype(np.float64) / n
    freqs_x1 = np.bincount(classes_x1, minlength=K_x).astype(np.float64) / n
    freqs_x2 = np.bincount(classes_x2, minlength=K_x).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / n
    nbins = np.array([K_x, K_x], dtype=np.int64)
    dtype = np.int32

    try:
        _ = compute_mi_from_classes(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
        )
    except Exception:
        pass
    try:
        _ = _count_nfailed_joint_indep_prange(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y,
            0.0, 2, 7, dtype,
        )
    except Exception:
        pass
    try:
        _ = _shuffle_and_compute_three_mis(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y_safe.copy(), freqs_y, dtype,
        )
    except Exception:
        pass

    # Conditional permutation shuffles.
    try:
        _conditional_shuffle_within_strata(classes_y_safe.copy(), classes_y, K_y)
    except Exception:
        pass
    try:
        _full_conditional_shuffle_ipf(
            classes_y_safe.copy(), classes_x1, classes_y, K_x, K_y,
        )
    except Exception:
        pass
    try:
        groups = rng.integers(0, 4, n).astype(np.int32)
        _group_aware_shuffle(classes_y_safe.copy(), groups, 4)
    except Exception:
        pass

    # Marginal + pair-search MI kernels (entry points for cat-FE search).
    factors_data = np.column_stack([classes_x1, classes_x2]).astype(dtype)
    marginal_mi = np.zeros(2, dtype=np.float64)
    try:
        _ = _marginal_screen_njit(
            factors_data, nbins, classes_y, freqs_y, dtype,
        )
    except Exception:
        pass
    pairs_a = np.array([0], dtype=np.int64)
    pairs_b = np.array([1], dtype=np.int64)
    try:
        _ = _pair_search_kernel_njit(
            factors_data, pairs_a, pairs_b, marginal_mi, nbins,
            classes_y, freqs_y, dtype,
        )
    except Exception:
        pass
    try:
        weights = np.ones(n, dtype=np.float64)
        _ = _pair_search_kernel_weighted_njit(
            factors_data, pairs_a, pairs_b, marginal_mi, nbins,
            classes_y, weights, dtype,
        )
    except Exception:
        pass

    # Discretisation kernels.
    cont = rng.normal(size=(n, 2)).astype(np.float64)
    try:
        _ = discretize_2d_array(cont, n_bins=4, method="quantile")
    except Exception:
        pass
    try:
        _ = discretize_array(cont[:, 0], n_bins=4, method="quantile")
    except Exception:
        pass

    _wall = time.perf_counter() - _t0
    if verbose:
        _log.info("prewarm_fs_numba_cache: warmed FS kernels in %.2fs", _wall)


__all__ = ["prewarm_fs_numba_cache"]
