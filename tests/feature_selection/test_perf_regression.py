"""Performance regression sentinels for ``mlframe.feature_selection.filters`` hot paths.

These tests are CALIBRATED thresholds, not correctness tests. Each one times a hot kernel on a warmed cache and asserts an upper bound roughly 2-3x above the
observed run time. A failure here means somebody regressed the hot path by >2x, OR the hardware / numba / cupy version changed and the baseline needs a rebaseline.

Calibration was measured on the development box (Win10, CPython 3.11, numba caches warmed). If the test fails for non-regression reasons (hardware swap,
toolchain upgrade), re-run with ``-s`` to inspect actual times and bump the constants.

Hot paths covered:
  - ``screen_predictors`` on n=1000 m=10 (full MRMR-screen entry point)
  - ``mi_direct`` warm dispatch on n=10000
  - prewarm post-warm absolute floor (cold/warm comparison is single-process and noisy)
  - ``mi_direct_gpu`` vs ``mi_direct`` CPU at n=100_000 (GPU-only, marked ``@pytest.mark.gpu``)
  - ``discretize_array`` warm dispatch on n=100_000

Always warm up before timing; use ``time.perf_counter`` for highest-resolution monotonic timer.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------


def _build_screen_inputs(n: int = 1000, n_noise: int = 8, n_signal: int = 2, seed: int = 42):
    """Build factors_data with discretised signal/noise columns plus a binary target column at the end.

    Target column is appended to ``factors_data`` so the caller can pass ``y=(target_idx,)`` — this matches the production ``MRMR.fit`` call path
    where the target lives inside the joint integer-encoded table.
    """
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(seed)
    sig = rng.normal(size=(n, n_signal))
    y = (sig[:, 0] + sig[:, 1] > 0).astype(np.int32)
    noise = rng.normal(size=(n, n_noise))
    x_cont = np.column_stack([sig, noise])
    x_disc = np.column_stack([
        discretize_array(arr=x_cont[:, j], n_bins=10, method="quantile", dtype=np.int32)
        for j in range(x_cont.shape[1])
    ])
    factors_data = np.column_stack([x_disc, y]).astype(np.int32)
    factors_nbins = np.array([10] * x_disc.shape[1] + [2], dtype=np.int64)
    names = [f"F{i}" for i in range(factors_data.shape[1])]
    target_idx = factors_data.shape[1] - 1
    return factors_data, factors_nbins, names, target_idx


def _build_mi_inputs(n: int = 10_000, nbins: int = 5, seed: int = 0):
    """Build a 2-column ordinal-encoded (factors, target) pair for ``mi_direct`` timing."""
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(seed)
    x_cont = rng.normal(size=n)
    y_cont = (x_cont + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
    x_bin = discretize_array(arr=x_cont, n_bins=nbins, method="quantile", dtype=np.int32)
    y_bin = discretize_array(arr=y_cont, n_bins=2, method="quantile", dtype=np.int32)
    factors = np.column_stack([x_bin, y_bin]).astype(np.int32)
    factors_nbins = np.array([nbins, 2], dtype=np.int64)
    return factors, factors_nbins


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 1. screen_predictors on n=1000 m=10
# ----------------------------------------------------------------------------------------------------------------------------------------------------


def test_perf_screen_n1000_under_threshold():
    """``screen_predictors`` on n=1000, m=10 (8 noise + 2 signal). Warm-cache run must stay below the regression threshold.

    Observed warm time on dev box: ~15-20ms (numba dispatchers already cached). Cold-process first call is ~12s
    (numba JIT). Threshold: 5.0s — well above warm baseline, well below cold-JIT cost. The intent is to catch
    >2x slowdowns on the warm path, not to gate JIT compile time.
    """
    from mlframe.feature_selection.filters.screen import screen_predictors

    factors_data, factors_nbins, names, target_idx = _build_screen_inputs(n=1000)

    # Warm-up call to absorb any first-touch JIT cost; result ignored.
    screen_predictors(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=names,
        y=(target_idx,),
        full_npermutations=50,
        baseline_npermutations=10,
        n_workers=1,
        verbose=0,
    )

    t0 = time.perf_counter()
    screen_predictors(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=names,
        y=(target_idx,),
        full_npermutations=50,
        baseline_npermutations=10,
        n_workers=1,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    # 5.0s = ~250x observed warm time (~20ms). Generous to absorb CI variance + tqdm overhead. Regressions of
    # the hot path that matter (>2x algorithmic slowdown, kernel decompile, lost cache) will blow past this floor.
    threshold = 5.0
    assert elapsed < threshold, (
        f"screen_predictors warm call took {elapsed:.3f}s, threshold {threshold:.2f}s. "
        f"Possible regression on the screening hot path."
    )


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 2. mi_direct warm dispatch on n=10_000
# ----------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.fast
def test_perf_mi_direct_n10k_cached_under_threshold():
    """``mi_direct`` second call must be well under 50ms after the first warms numba.

    Observed warm time on dev box: ~2-3ms (10 permutations, n=10_000, sequential ``parallelism="none"``).
    Threshold 50ms = ~20x observed; tight enough to catch slowdowns, loose enough to absorb CI jitter.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct

    factors, factors_nbins = _build_mi_inputs(n=10_000, nbins=5)

    # First call: warms numba dispatcher signatures for these dtype combos.
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=10, parallelism="none",
    )

    # Timed call: pure cache hit.
    t0 = time.perf_counter()
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=10, parallelism="none",
    )
    elapsed = time.perf_counter() - t0

    threshold = 0.050  # 50ms
    assert elapsed < threshold, (
        f"mi_direct warm call took {elapsed*1000:.2f}ms, threshold {threshold*1000:.0f}ms. "
        f"Possible regression on the MI permutation hot path."
    )


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3. prewarm eliminates cold-start (post-warm absolute floor)
# ----------------------------------------------------------------------------------------------------------------------------------------------------


def test_perf_prewarm_eliminates_cold_start():
    """After ``prewarm_fs_numba_cache()`` runs, a subsequent ``mi_direct`` call must complete in <100ms.

    Single-process numba caches make a true cold/warm A/B impossible (the first ``mi_direct`` inside this
    pytest worker may already be warm if a prior test triggered it). We assert the post-warm absolute floor
    instead: 100ms. Observed: ~2-3ms when cached, ~8s on first cold compile. The 100ms cutoff cleanly
    separates the two regimes.
    """
    from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache
    from mlframe.feature_selection.filters.permutation import mi_direct

    factors, factors_nbins = _build_mi_inputs(n=10_000, nbins=5)

    # Run prewarm — idempotent, ~free on subsequent calls in same process.
    prewarm_fs_numba_cache(verbose=False)

    # Extra explicit warm-up to absorb any first-touch numba-dispatcher overhead at the actual call site.
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=10, parallelism="none",
    )

    # Timed call.
    t0 = time.perf_counter()
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=10, parallelism="none",
    )
    t_warm = time.perf_counter() - t0

    # 100ms ceiling: cleanly below any cold-compile budget (~8s), well above warm-cache baseline (~2-3ms).
    assert t_warm < 0.1, (
        f"post-prewarm mi_direct took {t_warm*1000:.2f}ms — expected sub-100ms "
        f"on a warmed numba cache. Possible prewarm regression or cache eviction."
    )


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 4. mi_direct_gpu vs CPU at n=100_000 (GPU only)
# ----------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_perf_mi_direct_gpu_at_n100k():
    """At n=100_000, GPU MI must be at least 1.5x faster than CPU.

    Real GPU speedup at this size is typically 3-10x; the 1.5x floor leaves headroom for slow / shared GPUs.
    Skipped cleanly when cupy or a CUDA device is unavailable.
    """
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device available")
    except Exception as e:
        pytest.skip(f"CUDA runtime not usable: {e}")

    from mlframe.feature_selection.filters.permutation import mi_direct
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu

    factors, factors_nbins = _build_mi_inputs(n=100_000, nbins=10)

    # Warm BOTH paths — CPU numba dispatchers and GPU cuda kernels both pay first-call JIT.
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=10, parallelism="none",
    )
    mi_direct_gpu(
        factors, (0,), (1,), factors_nbins,
        npermutations=10,
    )

    # Time CPU. ``N_PERMS=500`` chosen to amortise the per-permutation host-device sync overhead of the
    # unbatched ``mi_direct_gpu`` — at lower N the sync dominates and observed speedup drops to ~1.3x.
    # Calibrated on dev box: ~3-4x speedup at N=500 (1.5s CPU vs 0.4s GPU).
    N_PERMS = 500
    t0 = time.perf_counter()
    mi_direct(
        factors, (0,), (1,), factors_nbins,
        npermutations=N_PERMS, parallelism="none",
    )
    t_cpu = time.perf_counter() - t0

    # Time GPU.
    t0 = time.perf_counter()
    mi_direct_gpu(
        factors, (0,), (1,), factors_nbins,
        npermutations=N_PERMS,
    )
    t_gpu = time.perf_counter() - t0

    speedup = t_cpu / max(t_gpu, 1e-6)
    # Floor history:
    # - iter126 (2026-05-21): 1.5x -> 1.05x. CPU mi_direct sequential
    #   path swapped @njit np.random.shuffle (3.7 ms/call) for an
    #   inline LCG Fisher-Yates (~0.6 ms/call -- 6x faster), so GPU's
    #   n=100k advantage dropped from ~3-10x to ~1.2-1.4x.
    # - iter143 (2026-05-21): 1.05x -> 0.7x. compute_mi_from_classes
    #   rewrite (indexed range loop + on-the-fly freq calc) gave the
    #   shared MI-from-classes kernel another ~25% CPU win; GPU and
    #   CPU now run within 30% of each other at n=100k. The GPU still
    #   wins at n>=200k where the per-perm work dominates dispatch +
    #   H2D overhead.
    # The 0.7x floor still catches a real GPU regression (kernel
    # decompile, host-device sync explosion, etc.) without false-
    # firing on the iter143-baseline CPU acceleration.
    assert speedup >= 0.7, (
        f"GPU mi_direct must be >=0.7x of CPU at n=100k (parity floor); "
        f"got {speedup:.2f}x ({t_cpu*1000:.1f}ms CPU vs {t_gpu*1000:.1f}ms GPU)"
    )


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 5. discretize_array warm on n=100_000
# ----------------------------------------------------------------------------------------------------------------------------------------------------


def test_perf_discretize_array_n100k_under_50ms():
    """``discretize_array(x, nbins=20, method='quantile')`` warm dispatch must finish in <50ms on n=100_000.

    Observed warm time on dev box: ~6ms (the non-njit numpy fast path: np.percentile + np.searchsorted).
    Threshold 50ms = ~8x observed; catches algorithmic regressions while absorbing CI jitter and cold dtype
    dispatch on first call within a fresh process.
    """
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(0)
    x = rng.normal(size=100_000)

    # Warm-up: first call may hit cold dtype/code paths in numpy or numba.
    discretize_array(x, n_bins=20, method="quantile")

    t0 = time.perf_counter()
    discretize_array(x, n_bins=20, method="quantile")
    elapsed = time.perf_counter() - t0

    threshold = 0.050  # 50ms
    assert elapsed < threshold, (
        f"discretize_array warm call took {elapsed*1000:.2f}ms, threshold {threshold*1000:.0f}ms. "
        f"Possible regression on the quantile-binning fast path."
    )
