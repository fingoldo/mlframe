"""GPU-vs-CPU crossover bench for ``dispatch_batch_pair_mi`` (perf loop iter139).

Iters 43-138 of the perf loop ran CPU-only (CUDA_VISIBLE_DEVICES=""), so the GPU dispatcher paths were never exercised in the loop. This bench
measures the REAL CPU(njit serial/parallel)-vs-GPU(numba.cuda / cupy) crossover for batch_pair_mi on the live GPU and checks whether the
hardcoded fallback thresholds (CUDA_MIN_ROWS=400_000, calibrated on a GTX 1050 Ti cc 6.1) route correctly here.

Measured 2026-06-15 on an NVIDIA RTX 500 Ada Generation Laptop GPU (~4GB VRAM, CUDA 12.9, cupy 14.1.1), n_pairs=64, best-of-7 warm:

    n          serial(ms)  par(ms)   cuda(ms)   winner   cuda|serial maxdiff
    10_000        1.26      1.24       9.06       par     6.9e-18
    75_000        9.62      ----      25.70       par     (CPU below crossover)
    100_000      14.11     13.87       4.89       cuda    1.1e-18   (2.79x vs par)
    150_000       ----     32.18       7.06       cuda    (4.56x vs par)
    200_000       ----     62.42      26.78       cuda    (~2.3x vs par)
    300_000       ----    114.55      17.74       cuda    (~6.5x vs par)
    400_000       ----    160.89      18.86       cuda    (~8.5x vs par)
    500_000     208.85    207.16      20.82       cuda    3.3e-19  (10x vs par)
    1_000_000   434.40    433.01      35.02       cuda    1.9e-19  (12x vs par)

Verdict (RESOLVED): the REAL CUDA crossover on this GPU is ~85-100k rows, NOT the hardcoded 400k. The hardcoded GTX-1050-Ti fallback
mis-routed the entire 100k-400k band to the CPU prange kernel, leaving 2.8x-8.5x on the table -- the same miscalibration class as the
2026-05-20 plug-in-MI incident. cuda is bit-identical to the CPU baseline (max_abs_diff ~1e-18..1e-19, pure FP reduction order).

Fix: NOT a new hardcoded constant. Populated the per-host kernel_tuning_cache via the registered ``_BPMI_SPEC`` sweep (``tune_spec``), and
lowered ``_BPMI_SWEEP_N_SAMPLES`` grid floor to {50_000, 100_000} so the cache learns the CPU-favorable low-n region instead of extrapolating
the lowest measured cell down to n=0 (which would mis-route 50-75k-row calls to a ~3x-slower GPU launch). The dispatcher already consults
``_BPMI_SPEC.choose()`` (cache) before the hardcoded fallback, so the measured regions now drive routing on this host.

Run:  PYTHONPATH=<worktree>/src python -m mlframe.feature_selection._benchmarks.bench_batch_pair_mi_gpu_crossover_iter139
"""
from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")


def _timeit(fn, args, reps: int = 7) -> float:
    fn(*args)  # warm (numba JIT + cupy NVRTC + first H2D)
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


def main() -> None:
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
        _CUDA_AVAIL,
        _CUPY_AVAIL,
        _make_batch_pair_mi_inputs,
        batch_pair_mi_cuda,
        batch_pair_mi_cupy,
        batch_pair_mi_njit_prange,
        batch_pair_mi_njit_serial,
    )

    if not _CUDA_AVAIL:
        print("numba.cuda not available -- nothing to bench.")
        return

    try:
        import cupy as cp
    except Exception:
        cp = None

    n_pairs = 64
    print(f"{'n':>10} {'serial':>9} {'par':>9} {'cuda':>9} {'cupy':>9}   winner  cuda|serial maxdiff")
    for n in (10_000, 75_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 1_000_000):
        args = _make_batch_pair_mi_inputs({"n_samples": n, "n_pairs": n_pairs})
        ts = _timeit(batch_pair_mi_njit_serial, args)
        tp = _timeit(batch_pair_mi_njit_prange, args)
        tc = _timeit(batch_pair_mi_cuda, args)
        tcp = _timeit(batch_pair_mi_cupy, args) if _CUPY_AVAIL else float("nan")
        ref = batch_pair_mi_njit_serial(*args)
        gc = batch_pair_mi_cuda(*args)
        maxdiff = float(np.max(np.abs(ref - gc)))
        cands = {"serial": ts, "par": tp, "cuda": tc, "cupy": tcp}
        win = min((v, k) for k, v in cands.items() if v == v)[1]
        print(f"{n:>10} {ts:>9.2f} {tp:>9.2f} {tc:>9.2f} {tcp:>9.2f}   {win:>6}  {maxdiff:.2e}")
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
