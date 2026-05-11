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
        # 2026-05-11 Wave 17: screening-path permutation kernels.
        # ``mi_direct`` (called once per candidate during screening) dispatches
        # to either ``parallel_mi_prange`` (parallel=True) or ``parallel_mi``
        # (sequential joblib worker) depending on caller. Both pay full
        # JIT-compile cost on first call (~17s on c0121 profile, dominating
        # MRMR.fit's wall time). Pre-warming shifts that out of the
        # user-visible timer.
        from .permutation import parallel_mi_prange, parallel_mi
        # 2026-05-11 Wave 17b: info-theory primitives used by screen.py.
        # ``merge_vars`` / ``entropy`` / ``mi`` / ``conditional_mi`` /
        # ``entropy_miller_madow`` each pay 1-3s JIT compile on first
        # call; ``screen_predictors`` calls them tens of times per fit.
        # Wave 17a covered permutation kernels (~1.5s saved on c0121);
        # the remaining ~16s of JIT compile attributed via
        # ``screen_predictors`` lives in these primitives.
        from .info_theory import (
            merge_vars, entropy, entropy_miller_madow, mi, conditional_mi,
        )
        from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
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

    # Info-theory primitives (Wave 17b). ``screen_predictors`` calls
    # ``merge_vars`` / ``entropy`` / ``mi`` / ``conditional_mi`` /
    # ``entropy_miller_madow`` for every candidate; the first call from
    # a fresh process eats ~5-8s of JIT compile (verified on c0121).
    # ``arr2str`` / ``count_cand_nbins`` / ``unpack_and_sort`` are
    # called from inside ``conditional_mi`` so they must compile too.
    try:
        # Realistic factors_data: 4 ordinal columns × n rows, dtype int32
        # to match the screening path's dtype.
        _factors = np.column_stack([
            classes_x1, classes_x2, classes_x1, classes_x2,
        ]).astype(np.int32)
        _nbins_4 = np.array([K_x, K_x, K_x, K_x], dtype=np.int64)
        # merge_vars on 2 vars
        _idx2 = np.array([0, 1], dtype=np.int64)
        _ = merge_vars(
            factors_data=_factors, vars_indices=_idx2, var_is_nominal=None,
            factors_nbins=_nbins_4, dtype=dtype,
        )
    except Exception:
        pass
    try:
        _f10 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        _ = entropy(freqs=_f10)
    except Exception:
        pass
    try:
        _f10 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        _ = entropy_miller_madow(freqs=_f10, n_samples=10)
    except Exception:
        pass
    try:
        _idxx = np.array([0], dtype=np.int64)
        _idxy = np.array([1], dtype=np.int64)
        _ = mi(_factors, _idxx, _idxy, _nbins_4, False, dtype)
    except Exception:
        pass
    try:
        _idxz = np.array([2], dtype=np.int64)
        _var_is_nom = np.zeros(4, dtype=np.bool_)
        _ = conditional_mi(
            _factors, _idxx, _idxy, _idxz, _var_is_nom, _nbins_4,
            -1.0, -1.0, -1.0, -1.0, None, False, False, dtype,
        )
    except Exception:
        pass
    try:
        _ = arr2str(_idx2)
    except Exception:
        pass
    try:
        _ = count_cand_nbins(_idx2, _nbins_4)
    except Exception:
        pass
    try:
        _ = unpack_and_sort(_idxx, _idxy)
    except Exception:
        pass

    # Screening permutation kernels (Wave 17). ``mi_direct`` calls one
    # of these per candidate during MRMR screening; first call on a
    # fresh process eats the entire ~17s JIT-compile budget. Prewarm
    # both the prange variant AND the joblib-worker variant so neither
    # caller path pays the cost during a real fit.
    try:
        _ = parallel_mi_prange(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=classes_y, freqs_y=freqs_y,
            npermutations=2, original_mi=0.0,
            base_seed=np.uint64(7), dtype=dtype,
        )
    except Exception:
        pass
    try:
        _ = parallel_mi(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=classes_y, freqs_y=freqs_y,
            npermutations=2, original_mi=0.0, max_failed=10, dtype=dtype,
        )
    except Exception:
        pass

    # Discretisation kernels. Cover the dominant production dtype combos:
    # int8 (default + cat-FE path) and int16 (categorize_dataset default
    # for the screening path on c0121-like regression combos). Each
    # dtype is a SEPARATE numba compilation (~9s of cold JIT compile,
    # surfaced 2026-05-11 on c0121 profile: 9.25s attributed to
    # ``categorize_dataset -> discretize_2d_array`` first call).
    # Wave 17c: explicitly pass ``min_values=None, max_values=None``
    # to match the categorize_dataset call signature exactly -- without
    # this, numba's per-signature cache misses and the kernel recompiles
    # on first real use.
    cont = rng.normal(size=(n, 2)).astype(np.float64)
    for _disc_dtype in (np.int8, np.int16):
        try:
            _ = discretize_2d_array(
                arr=cont, n_bins=4, method="quantile", min_ncats=50,
                min_values=None, max_values=None, dtype=_disc_dtype,
            )
        except Exception:
            pass
        try:
            _ = discretize_array(
                arr=cont[:, 0], n_bins=4, method="quantile",
                min_value=None, max_value=None, dtype=_disc_dtype,
            )
        except Exception:
            pass
    # Wave 17c: prewarm the inner-loop kernels directly. ``_discretize_array_impl``,
    # ``quantize_search``, ``quantize_dig``, ``discretize_uniform`` each compile
    # on first call from inside the prange body; prewarming the outer
    # ``discretize_2d_array`` only triggers them via the parallel-fanout
    # at runtime, which numba may not preserve in the disk cache cleanly.
    try:
        from .discretization import (
            _discretize_array_impl, quantize_search, quantize_dig,
            discretize_uniform, digitize, get_binning_edges,
        )
        _arr1d = cont[:, 0]
        for _disc_dtype in (np.int8, np.int16):
            try:
                _ = _discretize_array_impl(
                    arr=_arr1d, n_bins=4, method="quantile",
                    min_value=None, max_value=None, dtype=_disc_dtype,
                )
            except Exception:
                pass
            try:
                _ = discretize_uniform(
                    arr=_arr1d, n_bins=4, min_value=None, max_value=None,
                    dtype=_disc_dtype,
                )
            except Exception:
                pass
        _bins = np.linspace(_arr1d.min(), _arr1d.max(), 5)
        try:
            _ = quantize_search(_arr1d, _bins)
        except Exception:
            pass
        try:
            _ = quantize_dig(_arr1d, _bins)
        except Exception:
            pass
        try:
            _ = digitize(_arr1d, _bins, dtype=np.int32)
        except Exception:
            pass
        # get_binning_edges with both "quantile" and "uniform" method
        # branches -- numba compiles each branch separately because
        # the unicode_type narrows under the if/elif.
        for _method in ("quantile", "uniform"):
            try:
                _ = get_binning_edges(
                    arr=_arr1d, n_bins=4, method=_method,
                    min_value=None, max_value=None,
                )
            except Exception:
                pass
    except ImportError:
        pass

    _wall = time.perf_counter() - _t0
    if verbose:
        _log.info("prewarm_fs_numba_cache: warmed FS kernels in %.2fs", _wall)


__all__ = ["prewarm_fs_numba_cache"]
