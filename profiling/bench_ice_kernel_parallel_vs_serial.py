"""Bench + bit-identity check: ``_batch_per_class_ice_kernel`` (parallel-over-K) vs its serial twin
``_batch_per_class_ice_kernel_serial``, and the ``_ice_kernel_dispatch`` threshold they're wired behind.

Finding (2026-08-23): ``@numba.njit(parallel=True)`` with ``prange`` over K (number of classes) is
the wrong axis for the real calling pattern -- K is almost always 1-10 in production (binary=1,
multiclass rarely wide), so the fixed per-call thread-pool dispatch overhead (~10-40ms, independent
of trip count) never amortizes against K alone. It amortizes against the TOTAL work N*K instead.
Measured here (same-process A/B, warm, best-of-20-200):

    N=       569 K=1: parallel/serial ~955x SLOWER
    N=    15_000 K=1: parallel/serial ~155x SLOWER
    N=   100_000 K=1: parallel/serial  ~15x SLOWER
    N=    15_000 K=8: parallel/serial  ~10x SLOWER
    N=   300_000 K=1: parallel/serial   ~1.05x SLOWER (breakeven)
    N=   500_000 K=1: parallel/serial   ~1.32x SLOWER
    N= 1_000_000 K=1: parallel/serial   ~0.98x (breakeven)
    N= 1_000_000 K=4: parallel/serial   ~0.66x (1.5x FASTER)
    N=   500_000 K=8: parallel/serial   ~0.43x (2.3x FASTER)
    N= 1_000_000 K=8: parallel/serial   ~0.24x (4.2x FASTER)
    N= 5_000_000 K=8: parallel/serial   ~0.26x (3.8x FASTER)
    N= 2_000_000 K=20: parallel/serial  ~0.18x (5.5x FASTER)

Dispatch is via the per-host kernel_tuning_cache (``_ICE_KERNEL_PARALLELISM_SPEC``, no hardcoded
threshold) -- ``_ice_kernel_fallback_choice`` (used pre-sweep / on tuner failure) routes at
N*K=2_000_000, in the dead zone between the confirmed-loss region (N*K<=500k) and the confirmed-win
region (N*K~=4M), biased toward the serial side since the downside is asymmetric: an unnecessary
serial call at N*K~=1-2M costs a few ms, an unnecessary parallel call at small N*K costs 10-955x per
the ratios above.

Run: ``python profiling/bench_ice_kernel_parallel_vs_serial.py``
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.metrics.classification._classification_report import (
    _batch_per_class_ice_kernel,
    _batch_per_class_ice_kernel_serial,
    _ice_kernel_dispatch,
    _ice_kernel_fallback_choice,
)

_KWARGS = (10, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0, 0.0)


def _make_inputs(n: int, k: int, seed: int):
    """Build a synthetic (y_true_NK, y_pred_NK, desc_idx_NK) triple with realistic tie density."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, k)) > 0.5).astype(np.int8)
    # Round to 3 decimals to inject realistic tied-score density (quantized model outputs).
    y_pred = np.round(rng.random((n, k)), 3)
    desc_idx = np.ascontiguousarray(np.argsort(-y_pred, axis=0).astype(np.int64))
    return y_true, y_pred, desc_idx


def _bench(n: int, k: int, n_calls: int = 20) -> None:
    y_true, y_pred, desc_idx = _make_inputs(n, k, seed=0)
    _batch_per_class_ice_kernel(y_true, y_pred, desc_idx, *_KWARGS)
    _batch_per_class_ice_kernel_serial(y_true, y_pred, desc_idx, *_KWARGS)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_par = _batch_per_class_ice_kernel(y_true, y_pred, desc_idx, *_KWARGS)
    t_par = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_ser = _batch_per_class_ice_kernel_serial(y_true, y_pred, desc_idx, *_KWARGS)
    t_ser = (time.perf_counter() - t0) / n_calls

    max_diff = float(np.max(np.abs(out_par - out_ser)))
    print(f"N={n:>9} K={k:>3}: parallel={t_par*1000:8.3f}ms serial={t_ser*1000:8.3f}ms ratio={t_par/t_ser:6.2f}x max_diff={max_diff:.3e}")


def _check_bit_identity() -> None:
    """Exhaustive-ish bit-identity sweep: serial vs parallel must agree exactly (same per-class
    computation regardless of which thread runs it)."""
    max_seen = 0.0
    for n in (1, 2, 50, 500, 5000):
        for k in (1, 2, 3, 5, 10):
            for seed in range(3):
                y_true, y_pred, desc_idx = _make_inputs(n, k, seed)
                out_par = _batch_per_class_ice_kernel(y_true, y_pred, desc_idx, *_KWARGS)
                out_ser = _batch_per_class_ice_kernel_serial(y_true, y_pred, desc_idx, *_KWARGS)
                d = float(np.max(np.abs(out_par - out_ser))) if out_par.size else 0.0
                max_seen = max(max_seen, d)
                assert d == 0.0 or np.isnan(out_par).all(), f"n={n} k={k} seed={seed}: diff={d}"
    print(f"bit-identity sweep OK (5x5x3 = 75 scenarios), max diff = {max_seen}")

    # Dispatch threshold sanity (fallback path, used when the kernel_tuning_cache has no region yet):
    # below the cutoff routes to serial, at/above routes to parallel.
    below_n, below_k = 100, 1  # N*K = 100 << threshold
    above_n, above_k = 2_000_000, 1  # N*K == fallback threshold exactly
    y_true, y_pred, desc_idx = _make_inputs(below_n, below_k, seed=0)
    out_dispatch_below = _ice_kernel_dispatch(y_true, y_pred, desc_idx, *_KWARGS)
    out_serial_below = _batch_per_class_ice_kernel_serial(y_true, y_pred, desc_idx, *_KWARGS)
    assert np.array_equal(out_dispatch_below, out_serial_below, equal_nan=True)
    y_true, y_pred, desc_idx = _make_inputs(above_n, above_k, seed=0)
    out_dispatch_above = _ice_kernel_dispatch(y_true, y_pred, desc_idx, *_KWARGS)
    out_parallel_above = _batch_per_class_ice_kernel(y_true, y_pred, desc_idx, *_KWARGS)
    assert np.array_equal(out_dispatch_above, out_parallel_above, equal_nan=True)
    print("dispatch threshold routing OK")


if __name__ == "__main__":
    _check_bit_identity()
    for n, k in [(569, 1), (15_000, 1), (100_000, 1), (15_000, 8), (300_000, 1), (500_000, 1), (1_000_000, 1), (1_000_000, 4), (500_000, 8), (1_000_000, 8)]:
        _bench(n, k)
