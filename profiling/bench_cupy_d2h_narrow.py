"""Isolated microbench (Q2b): narrow the count-matrix D2H from int64 -> int32.

batch_mi_with_noise_gate_cupy D2Hs the (P1, total_size) int64 joint-count matrix
(``cp.asnumpy(tile_counts)``) -- the top GPU cost after OPT-D (~16% of scene wall,
the 1050 Ti is PCIe-bound). The per-cell count is bounded by n (rows) so it always
fits int32 for any realistic n (< 2^31). Casting to int32 on-device before the D2H
HALVES the transferred bytes and is BYTE-IDENTICAL downstream: _mi_from_counts_cpu
reads the counts as integers (fx += jc; jc * inv_n) and n << 2^31 so no value changes.

This bench compares: D2H int64 (current) vs on-device cast-to-int32 + D2H int32,
on the (P1, total_size) shapes the scene gate produces. Run ONE python at a time.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np

try:
    import cupy as cp
    from cupy._statistics.histogram import _bincount_kernel as _bk
    HAVE = True
except Exception as e:
    print("cupy unavailable:", e); HAVE = False


def _gpu_time(fn, repeats=10):
    fn(); cp.cuda.runtime.deviceSynchronize()
    best = 1e30
    for _ in range(repeats):
        t = time.perf_counter(); fn(); cp.cuda.runtime.deviceSynchronize()
        best = min(best, time.perf_counter() - t)
    return best


def bench(n, K, nbins=10, K_y=2, P1=4):
    total_size = nbins * K_y * K
    rng = np.random.default_rng(0)
    # device count matrix like the gate's tile_counts (values in [0, n])
    host_counts = rng.integers(0, n, size=(P1, total_size)).astype(np.intp)
    d_counts = cp.asarray(host_counts)

    out64 = np.empty((P1, total_size), dtype=np.int64)
    def d2h_int64():
        out64[:, :] = cp.asnumpy(d_counts)

    out32 = np.empty((P1, total_size), dtype=np.int32)
    def d2h_int32_cast():
        out32[:, :] = cp.asnumpy(d_counts.astype(cp.int32))

    t64 = _gpu_time(d2h_int64)
    t32 = _gpu_time(d2h_int32_cast)
    # byte-identity of the integer values
    d2h_int64(); d2h_int32_cast()
    ok = np.array_equal(out64, out32.astype(np.int64))
    print(f"  n={n} K={K} P1={P1} total_size={total_size} (matrix {P1}x{total_size})")
    print(f"    D2H int64 {t64*1e3:7.3f}ms | cast+D2H int32 {t32*1e3:7.3f}ms  "
          f"({t64/t32:4.2f}x)  values_identical={ok}")
    return ok


if __name__ == "__main__":
    if not HAVE:
        sys.exit(0)
    print("=== Q2b: count-matrix D2H narrowing int64 -> int32 (scene gate shapes) ===")
    ok = True
    for K in (150, 300, 600, 1200):
        for P1 in (1, 4):  # nperm+1
            ok &= bench(2407, K, P1=P1)
    print("\nALL VALUES IDENTICAL:", ok)
