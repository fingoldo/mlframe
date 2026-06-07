"""Isolated microbench (Q2): cupy bincount on the FE-MI flat-index workload.

The scene MRMR sampler shows ~26% of fit-wall in cupy ``bincount`` -- but that
is NOT the histogram kernel: ``cupy.bincount`` runs TWO synchronizing host-blocking
validations on every call (``(x < 0).any()`` non-negativity check + ``cupy.max(x)``
to size the output), even though batch_mi_with_noise_gate_cupy already KNOWS the
output size (``minlength = rows*total_size``) and constructs the indices to be
non-negative by construction. This bench measures the win from calling cupy's
internal ``_bincount_kernel`` directly into a pre-zeroed array of the known size,
skipping both syncs. Output counts are BYTE-IDENTICAL (same kernel) so the
downstream integer MI reduction is unchanged.

Run with ONE python process at a time (RAM/VRAM-tight box).
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
    print("cupy/_bincount_kernel unavailable:", e)
    HAVE = False


def _gpu_time(fn, *a, repeats=8):
    fn(*a); cp.cuda.runtime.deviceSynchronize()
    best = 1e30
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*a)
        cp.cuda.runtime.deviceSynchronize()
        best = min(best, time.perf_counter() - t)
    return best


def bench(n, K, nbins=10, K_y=2):
    """Mirror the flat-index shape batch_mi_with_noise_gate_cupy bincounts:
    one y-row tile -> (n*K,) flat int64 indices into rows*total_size slots."""
    rng = np.random.default_rng(0)
    total_size = nbins * K_y * K
    # flat indices in [0, total_size) like the real (base + y) construction
    idx = rng.integers(0, total_size, size=n * K).astype(np.int64)
    d_idx = cp.asarray(idx)
    minlen = total_size

    def via_bincount():
        return cp.bincount(d_idx, minlength=minlen)[:minlen]

    def via_kernel():
        b = cp.zeros((minlen,), dtype=np.intp)
        _bk(d_idx, b)
        return b

    # correctness
    b1 = cp.asnumpy(via_bincount())
    b2 = cp.asnumpy(via_kernel())
    ok = bool((b1 == b2).all())

    t_bc = _gpu_time(via_bincount)
    t_kn = _gpu_time(via_kernel)
    print(f"  n={n} K={K} nbins={nbins} K_y={K_y} total_size={total_size} flatlen={n*K}")
    print(f"    cp.bincount {t_bc*1e3:8.3f}ms   direct-kernel {t_kn*1e3:8.3f}ms   "
          f"speedup {t_bc/t_kn:5.2f}x   byte_identical={ok}")
    return ok


if __name__ == "__main__":
    if not HAVE:
        sys.exit(0)
    print("=== cupy bincount sync-elimination bench (scene FE-MI shapes) ===")
    ok = True
    # scene FE: n=2407; K (candidate cols per chunk) spans tens..few hundred; K_y=2 (binary y)
    for K in (50, 150, 300, 600, 1200):
        ok &= bench(2407, K)
    # with permutations the tile stacks (nperm+1) y-rows -> bigger flat index
    print("--- larger (multi-perm tile) ---")
    for K in (300, 600):
        ok &= bench(2407 * 4, K)  # ~4 y-rows worth in one tile
    print("\nALL BYTE-IDENTICAL:", ok)
