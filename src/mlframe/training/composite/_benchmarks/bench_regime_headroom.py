"""cProfile + A/B bench for the regime-headroom map on a production-shape input (1M rows / 10 bins).

Run: ``python -m mlframe.training.composite._benchmarks.bench_regime_headroom``

Two questions:
1. cProfile of ``regime_headroom_map`` at prod shape -> where does the wall go?
2. Grouped SSE reduction A/B: the shipped ``np.bincount`` path vs a fused ``numba.njit`` single-pass kernel. With only
   ~10 bins the bincount output arrays are tiny/cache-resident, so the njit kernel is within noise of bincount while
   adding a JIT-warm dependency -> bincount stays the DEFAULT. The intrinsic cost is the single ``np.argsort``-backed
   ``np.quantile`` + ``np.searchsorted`` binning (O(n log n)), which has no cheaper exact alternative. Verdict: no
   actionable speedup.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite._regime_headroom import regime_headroom_map

try:
    import numba

    @numba.njit(cache=True)
    def _bin_stats_njit(codes, w, y, raw, comp, lag, has_lag, k):
        W = np.zeros(k)
        sse_raw = np.zeros(k)
        sse_comp = np.zeros(k)
        sse_lag = np.zeros(k)
        for i in range(codes.shape[0]):
            b = codes[i]
            wi = w[i]
            yi = y[i]
            er = raw[i] - yi
            ec = comp[i] - yi
            W[b] += wi
            sse_raw[b] += wi * er * er
            sse_comp[b] += wi * ec * ec
            if has_lag:
                el = lag[i] - yi
                sse_lag[b] += wi * el * el
        return W, sse_raw, sse_comp, sse_lag

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _bin_stats_bincount(codes, w, y, raw, comp, lag, k):
    er = raw - y
    ec = comp - y
    W = np.bincount(codes, weights=w, minlength=k)
    sse_raw = np.bincount(codes, weights=w * er * er, minlength=k)
    sse_comp = np.bincount(codes, weights=w * ec * ec, minlength=k)
    sse_lag = np.bincount(codes, weights=w * (lag - y) ** 2, minlength=k)
    return W, sse_raw, sse_comp, sse_lag


def _make_data(n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    axis = rng.normal(0.0, 1.0, n)
    y = rng.normal(50.0, 5.0, n)
    raw = y + rng.normal(0.0, 1.0, n)
    comp = y + rng.normal(0.0, 0.6, n)
    lag = y + rng.normal(0.0, 1.5, n)
    return y, raw, comp, lag, axis


def _best_of(fn, *args, reps=7):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    y, raw, comp, lag, axis = _make_data()

    print("=== cProfile: regime_headroom_map (1M rows / 10 bins, with lag) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        regime_headroom_map(y, raw, comp, lag, axis_values=axis, n_bins=10)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())

    t_full, _ = _best_of(lambda: regime_headroom_map(y, raw, comp, lag, axis_values=axis, n_bins=10), reps=5)
    print(f"full map best-of-5: {1e3 * t_full:.2f} ms\n")

    print("=== bin SSE reduction A/B (n=1M, k=10) ===")
    k = 10
    codes = np.clip((axis - axis.min()) / (np.ptp(axis) + 1e-12) * k, 0, k - 1).astype(np.int64)
    w = np.ones_like(y)
    t_bc, _ = _best_of(_bin_stats_bincount, codes, w, y, raw, comp, lag, k)
    print(f"bincount (DEFAULT): {1e3 * t_bc:.2f} ms")
    if _HAVE_NUMBA:
        _bin_stats_njit(codes[:10], w[:10], y[:10], raw[:10], comp[:10], lag[:10], True, k)  # warm JIT
        t_nj, _ = _best_of(_bin_stats_njit, codes, w, y, raw, comp, lag, True, k)
        print(f"njit fused single-pass: {1e3 * t_nj:.2f} ms")
        print(f"njit speedup vs bincount: {t_bc / t_nj:.2f}x (near 1.0 -> no actionable win at k=10)")
    else:
        print("numba unavailable; bincount only")


if __name__ == "__main__":
    main()
