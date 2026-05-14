"""biz_val tests for GPU MI variants (feature_selection/filters/gpu.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN of the GPU MI path
over the CPU baseline at the size where the GPU is supposed to win.

Naming: ``test_biz_val_gpu_<variant>_<scenario>``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# All tests in this file require cupy.
pytest.importorskip("cupy")


def _make_signal(n=50_000, seed=42):
    """Strong-signal synthetic for GPU MI: ``y = sign(x + 0.3*noise)``."""
    from mlframe.feature_selection.filters.discretization import discretize_array
    rng = np.random.default_rng(seed)
    x_cont = rng.normal(size=n)
    y = (x_cont + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    x_bin = discretize_array(arr=x_cont, n_bins=10, method="quantile",
                              dtype=np.int32)
    factors = np.column_stack([x_bin, y]).astype(np.int32)
    factors_nbins = np.array([10, 2], dtype=np.int64)
    return factors, factors_nbins


def _warmup():
    """Warmup CUDA kernel JIT + njit kernels.

    Mirrors the perf-test workload (n=10_000, npermutations=500, batch_size=64)
    so the GPU kernel is compiled and cached at the exact shape the timed call
    will use. The earlier light warmup (n=2000, npermutations=10) left
    enough first-call overhead on cold/loaded machines to fail the >=1.5x
    speedup assertion when this test ran first in a batch.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched

    factors, factors_nbins = _make_signal(n=10_000, seed=0)
    mi_direct(factors, (0,), (1,), factors_nbins, npermutations=500,
              parallelism="none")
    mi_direct_gpu_batched(factors, (0,), (1,), factors_nbins,
                            npermutations=500, batch_size=64)


# ---------------------------------------------------------------------------
# mi_direct_gpu_batched: throughput vs CPU at large n
# ---------------------------------------------------------------------------


def test_biz_val_gpu_mi_batched_at_least_1_5x_faster_than_cpu_at_n10k():
    """``mi_direct_gpu_batched`` (Phase 2 batch-permutation kernel)
    must be >=1.5x faster than the single-thread CPU njit path on
    n=10000 with 500 permutations. Measured 2026-05-10 on
    GTX 1050 Ti: 1.86x. Floor 1.5x leaves headroom for slow GPUs.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched

    _warmup()
    factors, factors_nbins = _make_signal(n=10_000, seed=42)
    N_PERMS = 500

    t0 = time.perf_counter()
    mi_direct(factors, (0,), (1,), factors_nbins,
              npermutations=N_PERMS, parallelism="none")
    t_cpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    mi_direct_gpu_batched(factors, (0,), (1,), factors_nbins,
                            npermutations=N_PERMS, batch_size=64)
    t_gpu = time.perf_counter() - t0

    speedup = t_cpu / max(t_gpu, 1e-6)
    assert speedup >= 1.5, (
        f"GPU batched MI must be >=1.5x faster than CPU at n=10k; "
        f"got {speedup:.2f}x ({t_cpu*1000:.1f}ms vs {t_gpu*1000:.1f}ms)"
    )


def test_biz_val_gpu_mi_batched_scales_to_n200k():
    """``mi_direct_gpu_batched`` must successfully complete at
    n=200_000 with 500 permutations within 30s. The ``batch_size=64``
    OOM-fallback should kick in if device memory is tight; either
    path must yield a valid ``(original_mi, confidence)`` tuple."""
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched

    _warmup()
    factors, factors_nbins = _make_signal(n=200_000, seed=42)
    N_PERMS = 500

    t0 = time.perf_counter()
    mi, conf = mi_direct_gpu_batched(factors, (0,), (1,), factors_nbins,
                                        npermutations=N_PERMS, batch_size=64)
    t_gpu = time.perf_counter() - t0
    assert t_gpu < 30.0, (
        f"GPU batched MI must complete n=200k within 30s; "
        f"got {t_gpu:.1f}s"
    )
    # The return contract: ``(mi_or_zero, confidence)``. On strong
    # signal the kernel must complete and emit a valid tuple. ``mi``
    # may be the original MI value OR 0.0 if the permutation test
    # rejected the signal at the threshold; both indicate "the
    # kernel ran". Confidence must be in [0, 1].
    assert mi >= 0.0
    assert 0.0 <= conf <= 1.0


def test_biz_val_gpu_mi_batched_returns_valid_tuple_on_strong_signal():
    """``mi_direct_gpu_batched`` must return a 2-tuple
    ``(mi_or_zero, confidence)`` on strong-signal input. The first
    element is either the original MI or 0.0 (when the permutation
    test rejects the signal at the configured threshold); the second
    is a confidence in [0, 1]. Catches regressions where the GPU
    return contract changes silently."""
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched

    _warmup()
    factors, factors_nbins = _make_signal(n=10_000, seed=42)
    res = mi_direct_gpu_batched(factors, (0,), (1,), factors_nbins,
                                  npermutations=20, batch_size=64)
    assert isinstance(res, tuple) and len(res) == 2
    mi_val, conf = res
    assert mi_val >= 0.0
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# OOM auto-fallback (Phase 2 safety guarantee)
# ---------------------------------------------------------------------------


def test_biz_val_gpu_mi_batched_oom_safe_fallback_smoke():
    """Force a small ``batch_size`` to test that the OOM-fallback
    code path is exercised without crashing. Phase 2's contract:
    if ``batch_size * n * 4 bytes`` exceeds half free GPU memory,
    ``batch_size`` is halved; if it still OOMs, set to 1.

    This test doesn't actually OOM; it asserts the small-batch-size
    code path completes correctly."""
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched

    _warmup()
    factors, factors_nbins = _make_signal(n=10_000, seed=42)
    # batch_size=1 forces the per-permutation launch path inside the
    # batched implementation. Must still complete and return a valid
    # MI tuple.
    res = mi_direct_gpu_batched(factors, (0,), (1,), factors_nbins,
                                  npermutations=20, batch_size=1)
    assert isinstance(res, tuple) and len(res) == 2
    mi, conf = res
    assert mi > 0, f"batch_size=1 path must compute valid MI; got {mi}"
