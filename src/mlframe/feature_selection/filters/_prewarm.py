"""Numba-JIT pre-warm helpers for the feature_selection.filters package.

Calling ``prewarm_fs_numba_cache()`` once at process start triggers the JIT compile of all hot ``@njit`` kernels in:

  - ``cat_interactions``  (cat-FE permutation tests, pair search)
  - ``discretization``    (quantile / uniform binning)
  - ``info_theory``       (mutual-information primitives)

Without prewarm the first ``MRMR.fit`` call pays ~60 s of cumulative JIT cold-start. ``@njit(cache=True)`` writes the compiled object to disk only after a SUCCESSFUL first run, so cold-start latency hits every process that hasn't yet cached. Prewarm overlaps it with the suite's startup phase rather than the timed fit phase.

Lazy by design: kernels are imported only when ``prewarm_fs_numba_cache()`` is called, so importing ``mlframe.feature_selection.filters`` stays fast when the user doesn't need MRMR. Idempotent: numba caches compilations process-wide, so subsequent calls are essentially free.
"""

from __future__ import annotations

import numpy as np


def prewarm_fs_numba_cache(verbose: bool = False) -> None:
    """Trigger JIT compilation of feature_selection numba kernels.

    Safe to call from anywhere; no-op on numba-unavailable systems. Each kernel is exercised on a tiny synthetic input that matches its production signature.
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
        # Screening-path permutation kernels. ``mi_direct`` (called once per candidate during screening) dispatches to either ``parallel_mi_prange``
        # (parallel=True) or ``parallel_mi`` (sequential joblib worker); both pay full JIT-compile cost on first call (~17s).
        from .permutation import parallel_mi_prange, parallel_mi_prange_with_null, parallel_mi, shuffle_arr
        # Info-theory primitives used by screen.py. ``merge_vars`` / ``entropy`` / ``mi`` / ``conditional_mi`` / ``entropy_miller_madow`` each pay 1-3s JIT
        # compile on first call; ``screen_predictors`` calls them tens of times per fit.
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
            classes_x2, freqs_x2, classes_y_safe.copy(), freqs_y, 0, dtype,
        )
    except Exception:
        pass

    # Conditional permutation shuffles.
    try:
        _conditional_shuffle_within_strata(classes_y_safe.copy(), classes_y, K_y, 0)
    except Exception:
        pass
    try:
        _full_conditional_shuffle_ipf(
            classes_y_safe.copy(), classes_x1, classes_y, K_x, K_y, 0,
        )
    except Exception:
        pass
    try:
        groups = rng.integers(0, 4, n).astype(np.int32)
        _group_aware_shuffle(classes_y_safe.copy(), groups, 4, 0)
    except Exception:
        pass

    # Marginal + pair-search MI kernels (entry points for cat-FE search).
    factors_data = np.column_stack([classes_x1, classes_x2]).astype(dtype)
    marginal_mi = np.zeros(2, dtype=np.float64)
    candidate_idxs = np.arange(factors_data.shape[1], dtype=np.int64)
    try:
        _ = _marginal_screen_njit(
            factors_data, candidate_idxs, nbins, classes_y, freqs_y, dtype,
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

    # Info-theory primitives. ``screen_predictors`` calls ``merge_vars`` / ``entropy`` / ``mi`` / ``conditional_mi`` / ``entropy_miller_madow`` for every
    # candidate; first call from a fresh process eats ~5-8s of JIT compile. ``arr2str`` / ``count_cand_nbins`` / ``unpack_and_sort`` are called from inside
    # ``conditional_mi`` so they must compile too.
    try:
        # Realistic factors_data: 4 ordinal columns x n rows, dtype int32 to match the screening path's dtype.
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

    # Screening permutation kernels. ``mi_direct`` calls one of these per candidate during MRMR screening; first call on a fresh process eats the entire ~17s
    # JIT-compile budget. Prewarm both the prange variant AND the joblib-worker variant so neither caller path pays the cost during a real fit.
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
    # Null-mean-accumulating prange twin -- the kernel ``mi_direct(return_null_mean=True)`` runs for the MRMR relevance debiasing. Separate njit signature => separate compile.
    try:
        _ = parallel_mi_prange_with_null(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=classes_y, freqs_y=freqs_y,
            npermutations=2, original_mi=0.0,
            base_seed=np.uint64(7), dtype=dtype,
        )
    except Exception:
        pass
    # Prewarm ``shuffle_arr`` -- the njit wrapper around ``np.random.shuffle`` used in ``mi_direct``'s sequential fallback path (n_workers=1).
    try:
        _test = classes_y.copy()
        shuffle_arr(_test)
    except Exception:
        pass
    # Prewarm ``shuffle_arr_lcg`` -- inline-LCG Fisher-Yates added in iter126;
    # mi_direct's seq path now uses this instead of shuffle_arr for the 6x
    # speedup. Still prewarm the legacy shuffle_arr for any external caller.
    try:
        from .permutation import shuffle_arr_lcg
        _test_lcg = classes_y.copy()
        shuffle_arr_lcg(_test_lcg, np.uint64(42))
    except Exception:
        pass

    # Discretisation kernels. Cover the dominant production dtype combos: int8 (default + cat-FE path) and int16 (categorize_dataset default for the screening
    # path on regression combos). Each dtype is a SEPARATE numba compilation (~9s of cold JIT compile). Explicitly pass ``min_values=None, max_values=None`` to
    # match the categorize_dataset call signature exactly -- without this, numba's per-signature cache misses and the kernel recompiles on first real use.
    cont = rng.normal(size=(n, 2)).astype(np.float64)
    # polars ``.to_numpy()`` returns F-contiguous (column-major) while pandas returns C-contiguous (row-major). Numba's array-layout dispatch treats these as
    # DIFFERENT signatures (Array(float64, 2, 'C') vs Array(float64, 2, 'F')) and compiles separately -- prewarming only C-contig leaves the polars path with
    # ~10s of cold JIT compile on first MRMR.fit(polars_df) at 1M rows. Cover BOTH layouts.
    cont_f = np.asfortranarray(cont)  # F-contiguous variant
    # int32 is MRMR's DEFAULT quantization_dtype (see ``MRMR.__init__: quantization_dtype: object = np.int32`` at mrmr.py:294) -- prewarming only int8/int16
    # left the default-config code path paying ~8s of JIT compile per fresh process. int32 must be in the matrix.
    for _disc_dtype in (np.int8, np.int16, np.int32):
        for _arr in (cont, cont_f):
            try:
                _ = discretize_2d_array(
                    arr=_arr, n_bins=4, method="quantile", min_ncats=50,
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
    # Prewarm the inner-loop kernels directly. ``_discretize_array_impl``, ``quantize_search``, ``quantize_dig``, ``discretize_uniform`` each compile on first
    # call from inside the prange body; prewarming the outer ``discretize_2d_array`` only triggers them via the parallel-fanout at runtime, which numba may not
    # preserve in the disk cache cleanly.
    try:
        from .discretization import (
            _discretize_array_impl, quantize_search, quantize_dig,
            discretize_uniform, digitize, get_binning_edges,
        )
        _arr1d = cont[:, 0]
        for _disc_dtype in (np.int8, np.int16, np.int32):
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
        # get_binning_edges with both "quantile" and "uniform" method branches -- numba compiles each branch separately because the unicode_type narrows under the if/elif.
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

    # Layer-60 CMI-greedy plug-in entropy kernel. Called 1000+ times per MRMR
    # fit from _cmi_from_binned; warm it here so the first-fit JIT compile
    # (~0.3s) lands outside the timed screening path.
    try:
        from ._mi_greedy_cmi_fe import _entropy_from_classes_njit, _factorize_dense_njit
        _warm_cls = np.array([0, 1, 1, 2, 0], dtype=np.int64)
        _ = _entropy_from_classes_njit(_warm_cls)
        _ = _factorize_dense_njit(_warm_cls)
    except Exception:
        pass

    # Layer-90 numeric-decompose digit-extract kernel.
    try:
        from ._numeric_decompose_fe import _digit_extract_njit
        _ = _digit_extract_njit(np.array([1.23, -4.56, np.nan], dtype=np.float64), 100.0)
    except Exception:
        pass

    # Layer-95 periodic/modular kernel (warm all three op codes).
    try:
        from ._periodic_fe import _modular_njit, _modular_all_ops_njit
        _warm_mod = np.array([1.5, -2.5, np.nan, np.inf], dtype=np.float64)
        for _oc in (0, 1, 2):
            _ = _modular_njit(_warm_mod, 7.0, _oc)
        _ = _modular_all_ops_njit(_warm_mod, 7.0)
    except Exception:
        pass

    _wall = time.perf_counter() - _t0
    if verbose:
        _log.info("prewarm_fs_numba_cache: warmed FS kernels in %.2fs", _wall)


def prewarm_fs_cupy_kernels(verbose: bool = False) -> None:
    """Trigger CuPy RawKernel + sub-kernel compile-and-load for the GPU paths.

    Without this prewarm, the first ``mi_direct_gpu`` / ``mi_direct_gpu_batched``
    call on a fresh process pays ~1.65s of ``cupy.cuda.Module.load`` (29 NVRTC
    compile-loads measured in profile_mrmr_layer3_1m v6) -- the CuPy lazy-loaded
    sub-kernels for ``argsort``, ``sum``, ``random.uniform``, ``log``,
    elementwise compare, plus the project's ``compute_joint_hist_batched_cuda``
    and ``compute_mi_from_classes_cuda`` RawKernels.

    Calling this once at process start (alongside ``prewarm_fs_numba_cache``)
    moves the 1.65s out of the timed first-fit path. Subsequent fits hit the
    in-process CuPy module cache and pay zero.

    No-op on CuPy-unavailable / no-CUDA hosts. Idempotent: the second call
    completes in milliseconds because every kernel is already loaded.
    """
    import time
    import logging
    _log = logging.getLogger(__name__)
    _t0 = time.perf_counter()

    try:
        import cupy as cp  # noqa: F401 -- import doubles as the cupy-installed probe
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            if verbose:
                _log.info("prewarm_fs_cupy_kernels: CUDA unavailable; skipping")
            return
    except ImportError:
        if verbose:
            _log.info("prewarm_fs_cupy_kernels: cupy/pyutilz not importable; skipping")
        return
    except Exception:
        return

    try:
        from .gpu import (
            init_kernels, mi_direct_gpu, mi_direct_gpu_batched,
            mi_direct_gpu_batched_pairs,
        )
    except ImportError:
        return

    # Per Critic 2 D2: emit progress logs at INFO level regardless of
    # ``verbose`` so the 60-90s cold-start prewarm doesn't look like
    # a hang. Users who don't want the noise can filter the logger
    # ``mlframe.feature_selection.filters._prewarm`` at runtime.
    _log.info("prewarm_fs_cupy_kernels: building CuPy RawKernels ...")

    # Step 1: ensure the three project-owned RawKernels are built.
    try:
        init_kernels()
    except Exception:
        return

    # Step 2: tiny end-to-end mi_direct_gpu_batched call -- triggers the cupy
    # sub-kernel loads (argsort / sum / uniform / log / compare). npermutations
    # MUST be >= 32 to actually exercise the batched path (smaller goes to the
    # single-perm mi_direct_gpu loop). n=200 is enough to keep memory tiny
    # while still forcing each sub-kernel to load.
    try:
        rng = np.random.default_rng(0)
        n_warm = 200
        nbins = 3
        factors_data = rng.integers(0, nbins, size=(n_warm, 2)).astype(np.int32)
        factors_nbins = np.array([nbins, nbins], dtype=np.int32)
        _ = mi_direct_gpu_batched(
            factors_data=factors_data, x=(0,), y=(1,),
            factors_nbins=factors_nbins, npermutations=64, batch_size=32,
        )
    except Exception as _exc:
        _log.debug("prewarm_fs_cupy_kernels: batched warm failed (%s); continuing", _exc)

    # Step 3: tiny mi_direct_gpu call with npermutations<32 to warm the single-
    # perm path's distinct CuPy ops (cp.random.shuffle on a 1-D classes_y, the
    # per-iter compute_joint_hist_cuda + compute_mi_from_classes_cuda launches).
    try:
        _ = mi_direct_gpu(
            factors_data=factors_data, x=(0,), y=(1,),
            factors_nbins=factors_nbins, npermutations=4,
        )
    except Exception as _exc:
        _log.debug("prewarm_fs_cupy_kernels: single-perm warm failed (%s); continuing", _exc)

    # Step 4: tiny mi_direct_gpu_batched_pairs (cat_interactions GPU path) to
    # warm compute_joint_hist_multi_pair_cuda + per-pair joint-allocation kernels.
    try:
        rng2 = np.random.default_rng(1)
        n_pair_warm = 200
        n_feat = 3
        data_pair = rng2.integers(0, nbins, size=(n_pair_warm, n_feat)).astype(np.int32)
        nbins_pair = np.full(n_feat, nbins, dtype=np.int32)
        classes_y_pair = rng2.integers(0, nbins, size=n_pair_warm).astype(np.int32)
        freqs_y_pair = np.bincount(classes_y_pair, minlength=nbins).astype(np.float64) / n_pair_warm
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        _ = mi_direct_gpu_batched_pairs(
            factors_data=data_pair, pairs_a=pairs_a, pairs_b=pairs_b,
            factors_nbins=nbins_pair, classes_y=classes_y_pair, freqs_y=freqs_y_pair,
        )
    except Exception as _exc:
        _log.debug("prewarm_fs_cupy_kernels: batched_pairs warm failed (%s); continuing", _exc)

    # Step 4b: build the per-host kernel-tuning cache if missing. First-run
    # cost ~30s; subsequent processes load the cached JSON in ~1ms. The
    # cache persists at ``~/.pyutilz/kernel_tuning/{hw_fingerprint}.json``.
    _log.info("prewarm_fs_cupy_kernels: kernel_tuning sweep (~30s on first run, cached after) ...")
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.auto_tune import (
            ensure_joint_hist_tuning,
        )
        ensure_joint_hist_tuning(force=False)
    except Exception as _exc:
        _log.debug(
            "prewarm_fs_cupy_kernels: kernel_tuning auto-tune failed (%s); "
            "production will use hand-tuned fallbacks",
            _exc,
        )

    # Step 5: discretize_2d_array_cuda. The CuPy path uses cp.percentile +
    # cp.searchsorted + cp.linspace, each of which lazy-compiles a separate
    # CUDA module on first call. The percentile path alone accounts for ~700ms
    # of Module.load (measured in profile v7) when not pre-warmed. Use a frame
    # SIZE >= _DISCRETIZE_2D_CUDA_MIN_CELLS (default 500_000) so the dispatcher
    # actually routes to CUDA -- a smaller frame would hit the CPU njit path
    # and skip the warm.
    try:
        from .discretization import discretize_2d_array
        # 600 rows x 1000 cols = 600_000 cells, above the 500_000 dispatch
        # threshold; tiny in absolute terms (~5 MB float64).
        cont_warm = rng.normal(size=(600, 1000)).astype(np.float64)
        _ = discretize_2d_array(arr=cont_warm, n_bins=10, method="quantile")
    except Exception as _exc:
        _log.debug("prewarm_fs_cupy_kernels: discretize CUDA warm failed (%s); continuing", _exc)

    _wall = time.perf_counter() - _t0
    if verbose:
        _log.info("prewarm_fs_cupy_kernels: warmed CuPy kernels in %.2fs", _wall)


__all__ = ["prewarm_fs_numba_cache"]
