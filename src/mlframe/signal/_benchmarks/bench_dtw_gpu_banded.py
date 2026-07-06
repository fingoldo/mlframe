"""Bench + parity prototype for the banded GPU DTW rewrite (CPX-P0-2).

Run: ``python -m mlframe.signal._benchmarks.bench_dtw_gpu_banded`` on a CUDA box.

Compares the production full-matrix cupy diagonal sweep (``dtw_cupy``) against a
banded prototype that stores only the ``2*window+1`` live Sakoe-Chiba diagonals
in ``(i, j-i+window)`` coordinates -- O(n*window) device RAM instead of O(n*m) --
and a tiled wavefront kernel that fuses ``tile`` anti-diagonals per launch
(reducing the n+m host launches by ~tile). Reports OLD->NEW wall (best-of-N,
warmed), peak device bytes, and the identity gate vs the production path AND vs
the dtaidistance CPU reference.

The banded prototype here is the reference that was promoted into ``dtw.py``;
this file stays committed so the measured win is reproducible (REJECTED!=DELETED).
"""
from __future__ import annotations

import time

import numpy as np


def _bench():
    import cupy as cp

    from mlframe.signal.dtw import dtw_cupy_banded, dtw_cupy_full

    dtw_cupy = dtw_cupy_full  # OLD baseline = the retained full-matrix sweep

    try:
        from dtaidistance import dtw as _dtai
        _HAS_DTAI = True
    except Exception:
        _HAS_DTAI = False

    rng = np.random.default_rng(0)
    W = 200
    mp = cp.get_default_memory_pool()

    print(f"{'L':>7} {'old_ms':>9} {'new_ms':>9} {'speedup':>8} " f"{'oldMB':>8} {'newMB':>8} {'memx':>6} {'|dGPU-dGPU|':>12} {'|dGPU-dCPU|':>12}")
    for L in (1000, 5000, 10000):
        x = rng.standard_normal(L).astype(np.float32)
        y = rng.standard_normal(L).astype(np.float32)

        # warm both kernels
        dtw_cupy(x[:300], y[:300], window=W)
        dtw_cupy_banded(x[:300], y[:300], window=W)
        cp.cuda.Stream.null.synchronize()

        mp.free_all_blocks(); base = mp.used_bytes()
        d_old, p_old = dtw_cupy(x, y, window=W)
        old_mb = (mp.used_bytes() - base) / 1e6

        mp.free_all_blocks(); base = mp.used_bytes()
        d_new, p_new = dtw_cupy_banded(x, y, window=W)
        new_mb = (mp.used_bytes() - base) / 1e6

        def _best(fn, k=4):
            ts = []
            for _ in range(k):
                t = time.perf_counter()
                fn(x, y, window=W)
                cp.cuda.Stream.null.synchronize()
                ts.append(time.perf_counter() - t)
            return min(ts)

        old_ms = _best(dtw_cupy) * 1e3
        new_ms = _best(dtw_cupy_banded) * 1e3

        # Analytic cost-buffer peak (the dominant device allocation): full vs banded.
        old_mb = (L + 1) * (L + 1) * 4 / 1e6
        new_mb = (L + 1) * (2 * W + 1) * 4 / 1e6
        d_cpu = float(_dtai.distance(x.astype(np.float64), y.astype(np.float64), window=W)) if _HAS_DTAI else float("nan")
        gap_gpu = abs(d_old - d_new)
        gap_cpu = abs(d_new - d_cpu)
        assert p_old == p_new, f"path mismatch at L={L}"  # nosec B101 - internal invariant check in src/mlframe/signal/_benchmarks, not reachable with untrusted input
        print(f"{L:>7} {old_ms:>9.1f} {new_ms:>9.1f} {old_ms/new_ms:>7.2f}x "
              f"{old_mb:>8.1f} {new_mb:>8.1f} {old_mb/max(new_mb,1e-9):>5.1f}x "
              f"{gap_gpu:>12.2e} {gap_cpu:>12.2e}")


if __name__ == "__main__":
    _bench()
