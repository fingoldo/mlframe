"""Bench: FE-GPU batch executor free_blocks=True vs False.

The executor calls cp.get_default_memory_pool().free_all_blocks() after each VRAM column-chunk when
free_blocks=True (the default), forcing a real cudaFree + a fresh cudaMalloc on the next chunk instead of
cupy pool reuse - a genuine per-chunk cost, paid many times per fit and concurrently across devices under
multi_gpu_fe_batch_mi's ThreadPoolExecutor.

This bench measures the isolated per-chunk malloc/free churn both ways. It does NOT flip the default: the
flip is gated on a VRAM-safety leg that this single-GPU bench cannot supply - with free_blocks=False the
retained pool blocks on one thread's device are invisible to another thread's per-dispatch cushion check
(fe_gpu_has_vram_cushion), so pool retention on device A could starve device B's next cudaMalloc. Flip the
default only if BOTH (1) the wall improves end-to-end AND (2) ensure_fe_gpu_pool_limit provably bounds
retained VRAM under total-cushion on every device concurrently; otherwise gate free_blocks=(n_devices>1).

Requires a CUDA device (cupy). Skips cleanly otherwise.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_fe_batch_free_blocks
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import time


def main() -> None:
    try:
        import cupy as cp
    except ImportError:
        print("cupy unavailable - skipping (the free_blocks bench requires a CUDA device)")
        return

    n, k = 1_000_000, 512
    chunk = 64
    pool = cp.get_default_memory_pool()

    def _run(free_blocks: bool) -> float:
        pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _s in range(0, k, chunk):
            buf = cp.empty((n, chunk), dtype=cp.float64)
            buf += 1.0  # touch it so the alloc is real
            del buf
            if free_blocks:
                pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0

    _run(True)  # warm
    w_true = min(_run(True) for _ in range(3))
    w_false = min(_run(False) for _ in range(3))
    print(f"n={n} k={k} chunk={chunk} | free_blocks=True {w_true*1e3:8.2f}ms | free_blocks=False {w_false*1e3:8.2f}ms | x{w_true/max(1e-9,w_false):.2f}")
    print("NOTE: default NOT flipped -- see module docstring (multi-GPU VRAM-safety leg required before flipping).")


if __name__ == "__main__":
    main()
