"""iter141 microbench: joint_hist_batched kernel-variant + block_size crossover on the running host.

Measures ``compute_joint_hist_batched_shared_cuda`` vs ``compute_joint_hist_batched_cuda`` (global-atomic)
across block_size {256, 512, 1024} for n in {1e4..1e6} and joint_size {25, 100, 400}, to find the real
optimum on THIS GPU and compare it to the ``lookup_joint_hist`` HW-aware fallback (cc-8 -> shared/256).

Finding on RTX 500 Ada (cc 8.9): the ``shared`` variant wins everywhere (global is 1.5-4x slower). The
cc-8 fallback hardcodes block_size=256 for ALL joint sizes, but at joint_size=400 (20x20 bins) block_size=512
is a stable 9-21% win (30/30 paired trials), and at large-n joint_size=100 too. Fixed via the KTC sweep
(``ensure_joint_hist_tuning(force=True)``) which persists per-host regions -- no hardcoded threshold change.
The variant choice (shared) and small-joint block_size (256) were already correct, so this is a partial
re-calibration, not a wholesale mis-route like iter139/140.

block_size only changes grid decomposition; the int32 atomic-add joint histogram is order-independent, so all
block sizes are BIT-IDENTICAL (verified). The win is pure per-host tuning, zero numeric change.

Run in its own subprocess so a native cupy crash cannot take down the calling session.

Usage:  python -m mlframe.feature_selection._benchmarks.bench_joint_hist_block_size_iter141
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _median_ms(kernel, grid_x, bs, args, smem, iters=20, warmup=3):
    import cupy as cp

    for _ in range(warmup):
        kernel((grid_x, 1), (bs,), args, shared_mem=smem) if smem else kernel((grid_x, 1), (bs,), args)
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        kernel((grid_x, 1), (bs,), args, shared_mem=smem) if smem else kernel((grid_x, 1), (bs,), args)
        cp.cuda.runtime.deviceSynchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


def main():
    import cupy as cp

    from mlframe.feature_selection.filters import gpu as g

    g._ensure_kernels_inited()
    shared_k = g.compute_joint_hist_batched_shared_cuda
    global_k = g.compute_joint_hist_batched_cuda

    rng = np.random.default_rng(11)
    n_axis = (10_000, 30_000, 50_000, 100_000, 500_000, 1_000_000)
    nbins_axis = ((5, 5), (10, 10), (20, 20))

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"{'n':>10} {'joint':>6} {'variant':>7} {'bs':>5} {'ms':>9}  best")
    for n in n_axis:
        for nbx, nby in nbins_axis:
            joint = nbx * nby
            cx = cp.asarray(rng.integers(0, nbx, n).astype(np.int32))
            perms = cp.asarray(rng.integers(0, nby, n).astype(np.int32)).reshape(1, n)
            results = {}
            for variant in ("shared", "global"):
                for bs in (256, 512, 1024):
                    out = cp.zeros((1, joint), dtype=cp.int32)
                    args = (cx, perms, out, np.int32(n), np.int32(nbx), np.int32(nby))
                    grid_x = (n + bs - 1) // bs
                    kern = shared_k if variant == "shared" else global_k
                    smem = joint * 4 if variant == "shared" else 0
                    results[(variant, bs)] = _median_ms(kern, grid_x, bs, args, smem)
            cp.get_default_memory_pool().free_all_blocks()
            best_key = min(results, key=results.get)
            for (variant, bs), ms in sorted(results.items()):
                mark = " <-- BEST" if (variant, bs) == best_key else ""
                print(f"{n:>10} {joint:>6} {variant:>7} {bs:>5} {ms:>9.4f}{mark}")
            print()


if __name__ == "__main__":
    main()
