"""Bench: mi_direct_gpu_batched (serial single-stream) vs
``mi_direct_gpu_batched_streamed`` (2-stream pipelined).

Measures wall-clock per call at a few (n_rows, npermutations) shapes.
Min-of-N=5 iterations after a warm-up call, on the live GPU.

Expected on cc 6.1 (GTX 1050 Ti, 6 SMs): 0-15% gain (low SM count
saturates a single kernel; streams have little spare capacity to fill).
On cc 8.x (Ampere) the overlap should be larger.
"""
from __future__ import annotations

import time

import numpy as np


def _measure(fn, *args, n_iters: int = 5) -> float:
    import cupy as cp
    fn(*args)  # warm-up
    cp.cuda.runtime.deviceSynchronize()
    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(*args)
        cp.cuda.runtime.deviceSynchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    from mlframe.feature_selection.filters.gpu import (
        mi_direct_gpu_batched, mi_direct_gpu_batched_streamed,
    )

    print("=== streams vs serial bench ===")
    print(f"{'n_rows':>10} {'nperm':>6} {'serial_ms':>10} {'streamed_ms':>12} {'speedup':>8}")
    for n_rows in (200_000, 500_000, 1_000_000):
        for npermutations in (64, 128, 256):
            rng = np.random.default_rng(11)
            data = np.column_stack([
                rng.integers(0, 5, size=n_rows).astype(np.int32),
                rng.integers(0, 5, size=n_rows).astype(np.int32),
            ])
            nbins = np.array([5, 5], dtype=np.int32)
            try:
                t_serial = _measure(
                    mi_direct_gpu_batched, data, (0,), (1,), nbins,
                    npermutations, 64,
                )
                t_streamed = _measure(
                    mi_direct_gpu_batched_streamed, data, (0,), (1,), nbins,
                    npermutations, 64,
                )
            except Exception as e:
                print(f"  n={n_rows:>10_} nperm={npermutations:>6} ERROR: {type(e).__name__}: {e}")
                continue
            spd = t_serial / max(t_streamed, 1e-9)
            print(f"  {n_rows:>10_} {npermutations:>6} "
                  f"{t_serial * 1000:>9.2f}  {t_streamed * 1000:>11.2f}  {spd:>7.2f}x")


if __name__ == "__main__":
    main()
