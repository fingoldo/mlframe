"""A/B bench for the batched ``prange``-parallel bootstrap AUC resampler (2026-07-31).

Compares the current per-resample serial loop (``make_bootstrap_auc_resampler`` called once per
resample from Python, as ``bootstrap_metrics``'s serial path does) against
``bootstrap_auc_distribution_parallel`` -- a single ``numba.njit(parallel=True)`` call that runs every
resample's kernel body truly concurrently across OS threads, removing the per-resample Python/GIL
round-trip the module's own prior audit flagged as the real bottleneck (see
``evaluation/bootstrap.py``'s ``bootstrap_metrics`` docstring: "the per-resample cost is dominated by
the GIL-held index generation + fancy-index gather ... not the nogil kernels").

Confirms bit-identity (same RNG draw scheme) and measures wall-time at two shapes: n=500k (memory-safe
default) and n=2,000,000 (the profiled reporting/charts shape, smaller chunk to bound memory).

Usage:
    python profiling/bench_fused_resample_auc_batch_parallel.py
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.metrics._core_auc_brier import bootstrap_auc_distribution_parallel, make_bootstrap_auc_resampler


def _run(n: int, n_bootstrap: int, chunk_size: int, seed: int = 0):
    rng = np.random.default_rng(1)
    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.3).astype(np.int64)

    # Serial baseline: identical RNG draw scheme to bootstrap_auc_distribution_parallel's per-chunk draw,
    # but ONE resample at a time (matches bootstrap_metrics's serial per-resample np.random.default_rng loop).
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    serial_rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    serial_out = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = serial_rng.integers(0, n, size=n, dtype=np.int64)
        serial_out[i] = resampler(idx)
    t_serial = time.perf_counter() - t0

    # warm the parallel kernel's JIT compilation before timing (single small chunk)
    bootstrap_auc_distribution_parallel(y_true, y_score, n_bootstrap=4, random_state=999, chunk_size=4)

    t0 = time.perf_counter()
    parallel_out = bootstrap_auc_distribution_parallel(y_true, y_score, n_bootstrap=n_bootstrap, random_state=seed, chunk_size=chunk_size)
    t_parallel = time.perf_counter() - t0

    identical = np.array_equal(serial_out, parallel_out)
    print(f"n={n:,} n_bootstrap={n_bootstrap} chunk_size={chunk_size}")
    print(f"serial:   {t_serial:.4f}s")
    print(f"parallel: {t_parallel:.4f}s")
    print(f"speedup: {t_serial / t_parallel:.2f}x")
    print(f"bit-identical: {identical}")
    if not identical:
        ndiff = np.sum(serial_out != parallel_out)
        print(f"  MISMATCH: {ndiff}/{n_bootstrap} differ, max abs diff = {np.nanmax(np.abs(serial_out - parallel_out)):.3e}")
    print()
    return identical


def main():
    import numba

    print(f"numba threads available: {numba.config.NUMBA_NUM_THREADS}")
    print()
    ok1 = _run(n=500_000, n_bootstrap=1000, chunk_size=200)
    ok2 = _run(n=2_000_000, n_bootstrap=200, chunk_size=50)
    assert ok1 and ok2, "batched-parallel resampler must be bit-identical to the serial per-resample loop"


if __name__ == "__main__":
    main()
