"""Microbench: is plugin_mi_classif_batch_dispatch's cProfile tottime the njit
compute floor, or plain-python wrapper overhead?

The 1M-wellbore .prof attributes 65.6s tottime to
`_hermite_fe_mi.py:plugin_mi_classif_batch_dispatch`. Per the CLAUDE.md cProfile
caveat, @njit(parallel=True) body time is mis-attributed to the plain-python
CALLER frame. This bench separates:

  (A) full dispatch call  (python routing + njit kernel)
  (B) direct njit kernel  (the compute floor)
  (C) wrapper-only overhead  = A - B  (env reads, cached imports, gpu_globally_disabled)

at the representative FE-screen shape (n=30k, plugin_n_bins=20, many candidate
columns) and across the real k-distribution (many small-k calls + a few big-k).

Run (CPU-only):
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_plugin_mi_batch_dispatch_overhead

VERDICT (2026-07-06, CPU njit, n=30k, plugin_n_bins=20): COMPUTE-FLOOR REJECT.
The 65.6s .prof tottime is genuine @njit(parallel=True) per-column argsort-binning
plug-in MI compute, mis-attributed by cProfile to the plain-python dispatch caller
frame (numba's compiled C body has no python frame, so its time rolls into the
caller's tottime -- the documented CLAUDE.md caveat). The dispatch wrapper's OWN
routing (2 sys.modules-cached `from..import`, 2 os.environ.get, gpu_globally_disabled,
shape unpack) measures 10.1 us/call in isolation -> 1.6ms over the real ~157-call F2
count = 0.075% of the 2125ms njit compute; the full-vs-njit A/B aggregate is -0.78%
(noise, oscillates around 0). NOT wrapper-overhead, NOT wasted-work, NOT hoistable at
these shapes -- the kernel is at the memory-bandwidth floor (see the stacked
bench-rejected notes in hermite_fe/__init__.py:_plugin_mi_classif_batch_njit).
Below the 0.5% ship threshold -> no change shipped.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_MI_BACKEND", "njit")

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import (
    plugin_mi_classif_batch_dispatch,
    _plugin_mi_classif_batch_njit,
)


def _best(fn, *a, reps=7):
    fn(*a)  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _bench_shape(n, k, n_bins=20, seed=0, reps=7):
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((n, k)))
    y = np.ascontiguousarray((rng.random(n) < 0.3).astype(np.int64))
    t_full = _best(plugin_mi_classif_batch_dispatch, X, y, n_bins, reps=reps)
    t_njit = _best(_plugin_mi_classif_batch_njit, X, y, n_bins, reps=reps)
    over = t_full - t_njit
    # sanity: identical output
    a = plugin_mi_classif_batch_dispatch(X, y, n_bins)
    b = _plugin_mi_classif_batch_njit(X, y, n_bins)
    maxdiff = float(np.max(np.abs(a - b)))
    return t_full, t_njit, over, maxdiff


def main():
    print("plugin_mi_classif_batch_dispatch overhead microbench (CPU njit)")
    print(f"MLFRAME_MI_BACKEND={os.environ.get('MLFRAME_MI_BACKEND')!r} " f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print()
    print(f"{'n':>8} {'k':>5} {'full_ms':>9} {'njit_ms':>9} {'wrap_us':>9} {'wrap_%':>7} {'maxdiff':>9}")
    shapes = [
        (30000, 1), (30000, 5), (30000, 14), (30000, 30),
        (30000, 80), (30000, 306), (30000, 527),
    ]
    for n, k in shapes:
        t_full, t_njit, over, md = _bench_shape(n, k)
        print(f"{n:>8} {k:>5} {t_full*1e3:>9.3f} {t_njit*1e3:>9.3f} " f"{over*1e6:>9.1f} {100*over/t_full:>6.2f}% {md:>9.1e}")

    # Aggregate wrapper cost over a realistic call-count distribution (from
    # _orth_mi_backends note: warm F2 100k had ~157 dispatch calls; scale-agnostic
    # k-mix here to show per-call overhead * count).
    print()
    print("Per-call wrapper overhead x realistic call count (157 calls, k-mix from prod note):")
    kmix = [(1, 96), (14, 53), (40, 4), (306, 1), (527, 2), (80, 1)]
    total_over = 0.0
    total_full = 0.0
    for k, cnt in kmix:
        t_full, t_njit, over, _ = _bench_shape(30000, k, reps=5)
        total_over += over * cnt
        total_full += t_full * cnt
        print(f"  k={k:>4} x{cnt:>3}: wrap {over*1e6:>7.1f}us/call -> {over*cnt*1e3:>7.2f}ms")
    print(f"  TOTAL: wrapper {total_over*1e3:.2f}ms of full {total_full*1e3:.2f}ms " f"= {100*total_over/total_full:.2f}%")


if __name__ == "__main__":
    main()
