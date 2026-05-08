"""Benchmark GPU vs CPU implementations of mlframe.metrics batch helpers.

Runs four sweeps:

1. Backend-of-RMSE: cupy expression vs cupy ReductionKernel vs
   numba.cuda kernel. Justifies why mlframe ships ReductionKernel.

2. RMSE / ROC AUC / PR AUC: GPU (cupy via mlframe) vs CPU (numpy /
   numba per-col loop / sklearn) across (N, M) grid. Verifies the
   ~100k-row crossover and the 4-19x speedups documented in
   ``mlframe/metrics.py``.

3. Tie correctness: confirms ``gpu_multiple_roc_auc_scores`` matches
   sklearn bit-for-bit on heavily-tied scores (the naive
   ``argsort(argsort)`` snippet drifts ~1e-5 here).

4. End-to-end via dispatchers ``compute_batch_aucs`` /
   ``compute_batch_rmse`` with auto and forced backends, plus the
   ``set_gpu_thresholds`` knob.

Run::

    python bench_gpu_metrics.py            # full sweep
    python bench_gpu_metrics.py --quick    # smaller sizes, ~30s
    python bench_gpu_metrics.py --nbcuda   # also run numba.cuda comparison

Skips GPU sweeps gracefully if cupy / CUDA aren't present.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mlframe.metrics import (
    compute_batch_aucs,
    compute_batch_rmse,
    fast_aucs,
    fast_roc_auc,
    is_gpu_metrics_available,
    set_gpu_thresholds,
)

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


# -----------------------------------------------------------------------------
# Bench harness
# -----------------------------------------------------------------------------


def time_op(fn: Callable, *args, repeats: int = 3, warmup: int = 1, sync_gpu: bool = False):
    """Min-of-N timing. ``sync_gpu`` adds cuda stream sync per repeat."""
    for _ in range(warmup):
        fn(*args)
        if sync_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync_gpu and _HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
        timings.append(time.perf_counter() - t0)
    return out, min(timings)


def _free_gpu():
    if _HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()


# -----------------------------------------------------------------------------
# CPU references
# -----------------------------------------------------------------------------


def cpu_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    return np.sqrt(np.mean((y_true - y_pred) ** 2.0, axis=0))


def cpu_loop_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    M = y_score.shape[1]
    return np.array([fast_roc_auc(y_true, y_score[:, j]) for j in range(M)])


def cpu_loop_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    M = y_score.shape[1]
    out = np.empty(M)
    for j in range(M):
        _, ap = fast_aucs(y_true, y_score[:, j])
        out[j] = ap
    return out


def sklearn_loop_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    M = y_score.shape[1]
    return np.array([average_precision_score(y_true, y_score[:, j]) for j in range(M)])


# -----------------------------------------------------------------------------
# Sweep 1: RMSE backend comparison (cupy expr / cupy ReductionKernel / numba.cuda)
# -----------------------------------------------------------------------------


def sweep_rmse_backends(quick: bool, run_nbcuda: bool):
    if not _HAS_CUPY:
        print("\n[skip] sweep_rmse_backends: cupy not installed.")
        return
    print("=" * 80)
    print("Sweep 1: RMSE backends (cupy expr vs ReductionKernel vs numba.cuda)")
    print("=" * 80)
    sizes = [(100_000, 5), (1_000_000, 5), (1_000_000, 20)]
    if not quick:
        sizes.append((5_000_000, 5))

    cupy_red_kernel = cp.ReductionKernel(
        in_params="float64 y, float64 p",
        out_params="float64 z",
        map_expr="(y - p) * (y - p)",
        reduce_expr="a + b",
        post_map_expr="z = a",
        identity="0.0",
        name="bench_sse_per_col",
    )

    def cupy_expr(y_dev, p_dev):
        if y_dev.ndim == 1:
            y_dev = y_dev[:, cp.newaxis]
        return cp.sqrt(cp.mean((y_dev - p_dev) ** 2.0, axis=0))

    def cupy_reduction(y_dev, p_dev):
        if y_dev.ndim == 1:
            y_dev = y_dev[:, cp.newaxis]
        N = p_dev.shape[0]
        return cp.sqrt(cupy_red_kernel(y_dev, p_dev, axis=0) / N)

    nb_kernel = None
    if run_nbcuda:
        try:
            from numba import cuda

            @cuda.jit
            def _rmse_partial(y, p, partial, N, M):
                j = cuda.blockIdx.y
                i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
                if j >= M or i >= N:
                    return
                d = y[i] - p[i, j]
                cuda.atomic.add(partial, (cuda.blockIdx.x, j), d * d)

            def numba_rmse(y_dev, p_dev):
                N, M = p_dev.shape
                BLOCK = 256
                grid = (N + BLOCK - 1) // BLOCK
                partial = cp.zeros((grid, M), dtype=cp.float64)
                _rmse_partial[(grid, M), BLOCK](
                    cuda.as_cuda_array(y_dev),
                    cuda.as_cuda_array(p_dev),
                    cuda.as_cuda_array(partial),
                    N, M,
                )
                return cp.sqrt(cp.sum(partial, axis=0) / N)

            nb_kernel = numba_rmse
        except Exception as e:
            print(f"[skip] numba.cuda comparison: {e}")
            nb_kernel = None

    print(f"{'N':>9} {'M':>3} | {'expr':>9} | {'reduction':>10} | "
          f"{'numba':>9} | red/expr  numba/expr")
    print("-" * 80)
    for N, M in sizes:
        rng = np.random.default_rng(0)
        y = rng.standard_normal(N).astype(np.float64)
        p = rng.standard_normal((N, M)).astype(np.float64)
        y_dev = cp.asarray(y)
        p_dev = cp.asarray(p)

        _, t_expr = time_op(cupy_expr, y_dev, p_dev, sync_gpu=True)
        _, t_red = time_op(cupy_reduction, y_dev, p_dev, sync_gpu=True)
        if nb_kernel is not None:
            _, t_nb = time_op(nb_kernel, y_dev, p_dev, sync_gpu=True)
            nb_ratio = f"{t_nb/t_expr:5.2f}x"
            nb_str = f"{t_nb*1000:7.2f}ms"
        else:
            t_nb = None
            nb_ratio = "    n/a"
            nb_str = "    n/a"
        print(f"{N:>9} {M:>3} | {t_expr*1000:7.2f}ms | {t_red*1000:8.2f}ms | "
              f"{nb_str} | {t_red/t_expr:5.2f}x   {nb_ratio}")
        _free_gpu()


# -----------------------------------------------------------------------------
# Sweep 2: GPU vs CPU on the three batch metrics
# -----------------------------------------------------------------------------


def sweep_gpu_vs_cpu(quick: bool):
    print("\n" + "=" * 80)
    print("Sweep 2: GPU (mlframe) vs CPU (numba/numpy/sklearn) per metric")
    print("=" * 80)

    if not _HAS_CUPY:
        print("[skip] cupy not installed; CPU-only run not interesting in this script.")
        return

    sizes = [(10_000, 1), (10_000, 5), (100_000, 5), (100_000, 20),
             (1_000_000, 5), (1_000_000, 20)]
    if not quick:
        sizes.append((5_000_000, 5))

    print(f"\n--- RMSE ---")
    print(f"{'N':>9} {'M':>3} | {'cpu_np':>10} | {'gpu_disp':>10} | {'speedup':>8} | {'max_err':>10}")
    print("-" * 70)
    for N, M in sizes:
        rng = np.random.default_rng(0)
        y = rng.standard_normal(N).astype(np.float64)
        p = rng.standard_normal((N, M)).astype(np.float64)

        cpu_v, cpu_t = time_op(cpu_rmse, y, p)
        gpu_v, gpu_t = time_op(compute_batch_rmse, y, p,
                               sync_gpu=True,
                               # force GPU even below threshold so we benchmark
                               # the actual GPU path; the dispatcher would pick
                               # CPU at N<100k.
                               )
        # Re-run with force_backend='gpu' to get real GPU timing on small sizes
        def _gpu(yt, yp):
            return compute_batch_rmse(yt, yp, force_backend="gpu")
        gpu_v, gpu_t = time_op(_gpu, y, p, sync_gpu=True)
        max_err = float(np.max(np.abs(cpu_v - gpu_v)))
        print(f"{N:>9} {M:>3} | {cpu_t*1000:8.2f}ms | {gpu_t*1000:8.2f}ms | "
              f"{cpu_t/gpu_t:7.2f}x | {max_err:.2e}")
        _free_gpu()

    print(f"\n--- ROC AUC ---")
    print(f"{'N':>9} {'M':>3} | {'cpu_loop':>10} | {'sklearn':>10} | {'gpu_disp':>10} | {'speedup':>8} | {'max_err':>10}")
    print("-" * 90)
    for N, M in sizes:
        rng = np.random.default_rng(11)
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M)).astype(np.float64)
        repeats = 1 if N >= 1_000_000 else 3

        cpu_v, cpu_t = time_op(cpu_loop_roc_auc, y, p, repeats=repeats)
        if N <= 1_000_000:
            sk_v, sk_t = time_op(
                lambda yt, yp: np.array([roc_auc_score(yt, yp[:, j]) for j in range(yp.shape[1])]),
                y, p, repeats=repeats,
            )
            sk_str = f"{sk_t*1000:8.2f}ms"
        else:
            sk_v, sk_str = cpu_v, "  skipped"

        def _gpu_roc(yt, yp):
            return compute_batch_aucs(yt, yp, force_backend="gpu")[0]
        gpu_v, gpu_t = time_op(_gpu_roc, y, p, sync_gpu=True, repeats=repeats)
        max_err = float(np.max(np.abs(sk_v - gpu_v)))
        print(f"{N:>9} {M:>3} | {cpu_t*1000:8.2f}ms | {sk_str:>10} | "
              f"{gpu_t*1000:8.2f}ms | {cpu_t/gpu_t:7.2f}x | {max_err:.2e}")
        _free_gpu()

    print(f"\n--- PR AUC ---")
    print(f"{'N':>9} {'M':>3} | {'cpu_loop':>10} | {'sklearn':>10} | {'gpu_disp':>10} | {'speedup':>8} | {'max_err':>10}")
    print("-" * 90)
    for N, M in sizes:
        rng = np.random.default_rng(22)
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M)).astype(np.float64)
        repeats = 1 if N >= 1_000_000 else 3

        cpu_v, cpu_t = time_op(cpu_loop_pr_auc, y, p, repeats=repeats)
        if N <= 1_000_000:
            sk_v, sk_t = time_op(sklearn_loop_pr_auc, y, p, repeats=repeats)
            sk_str = f"{sk_t*1000:8.2f}ms"
        else:
            sk_v, sk_str = cpu_v, "  skipped"

        def _gpu_pr(yt, yp):
            return compute_batch_aucs(yt, yp, force_backend="gpu")[1]
        gpu_v, gpu_t = time_op(_gpu_pr, y, p, sync_gpu=True, repeats=repeats)
        max_err = float(np.max(np.abs(sk_v - gpu_v)))
        print(f"{N:>9} {M:>3} | {cpu_t*1000:8.2f}ms | {sk_str:>10} | "
              f"{gpu_t*1000:8.2f}ms | {cpu_t/gpu_t:7.2f}x | {max_err:.2e}")
        _free_gpu()


# -----------------------------------------------------------------------------
# Sweep 3: Tie correctness for ROC AUC
# -----------------------------------------------------------------------------


def sweep_tie_correctness():
    if not _HAS_CUPY:
        print("\n[skip] sweep_tie_correctness: cupy not installed.")
        return
    print("\n" + "=" * 80)
    print("Sweep 3: ROC AUC tie correctness (gpu avg-rank vs sklearn)")
    print("=" * 80)
    cases = [
        ("continuous", 50_000, lambda N, r: r.standard_normal(N)),
        ("10 bins (heavy ties)", 50_000,
         lambda N, r: r.choice(np.linspace(0, 1, 10), N)),
        ("100 bins", 50_000,
         lambda N, r: r.choice(np.linspace(0, 1, 100), N)),
    ]
    print(f"{'case':>26} | {'max_err vs sklearn':>20}")
    print("-" * 55)
    for label, N, gen in cases:
        rng = np.random.default_rng(33)
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = np.column_stack([gen(N, rng) for _ in range(5)])
        sk = np.array([roc_auc_score(y, p[:, j]) for j in range(5)])
        gpu = compute_batch_aucs(y, p, force_backend="gpu")[0]
        err = float(np.max(np.abs(sk - gpu)))
        print(f"{label:>26} | {err:.4e}")


# -----------------------------------------------------------------------------
# Sweep 4: dispatcher behavior
# -----------------------------------------------------------------------------


def sweep_dispatch():
    print("\n" + "=" * 80)
    print("Sweep 4: dispatcher behavior")
    print("=" * 80)
    print(f"is_gpu_metrics_available(): {is_gpu_metrics_available()}")

    rng = np.random.default_rng(44)
    y = (rng.standard_normal(50_000) > 0).astype(np.int8)
    p = rng.standard_normal((50_000, 4))

    print("\nBelow threshold (N=50k, threshold=100k) -> auto picks CPU:")
    _, t_auto = time_op(compute_batch_aucs, y, p)
    _, t_cpu = time_op(lambda yt, yp: compute_batch_aucs(yt, yp, force_backend="cpu"), y, p)
    if _HAS_CUPY:
        _, t_gpu = time_op(lambda yt, yp: compute_batch_aucs(yt, yp, force_backend="gpu"),
                           y, p, sync_gpu=True)
        print(f"  auto = {t_auto*1000:.2f}ms, force_cpu = {t_cpu*1000:.2f}ms, "
              f"force_gpu = {t_gpu*1000:.2f}ms")
    else:
        print(f"  auto = {t_auto*1000:.2f}ms, force_cpu = {t_cpu*1000:.2f}ms")

    print("\nLowering threshold so auto picks GPU at this size:")
    set_gpu_thresholds(n=10_000)
    try:
        _, t_auto2 = time_op(compute_batch_aucs, y, p, sync_gpu=True)
        print(f"  auto with N>=10k = {t_auto2*1000:.2f}ms")
    finally:
        set_gpu_thresholds(n=100_000)  # restore


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Smaller grid (~30s).")
    ap.add_argument("--nbcuda", action="store_true",
                    help="Also benchmark numba.cuda RMSE kernel for comparison.")
    args = ap.parse_args()

    if _HAS_CUPY:
        try:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            mem_gb = cp.cuda.runtime.memGetInfo()[1] / 1024**3
            print(f"GPU: {name}  ({mem_gb:.1f} GB)")
            print(f"cupy: {cp.__version__}")
        except Exception as e:
            print(f"cupy installed but no CUDA device visible: {e}")
    print(f"numpy: {np.__version__}")
    print(f"GPU metrics available: {is_gpu_metrics_available()}")

    sweep_rmse_backends(quick=args.quick, run_nbcuda=args.nbcuda)
    sweep_gpu_vs_cpu(quick=args.quick)
    sweep_tie_correctness()
    sweep_dispatch()


if __name__ == "__main__":
    sys.exit(main())
