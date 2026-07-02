"""cProfile + A/B bench for the composite VALUE report on a production-shape input (1M rows / 773 groups).

Run: ``python -m mlframe.training.composite._benchmarks.bench_composite_value_report``

Two questions:
1. cProfile of ``build_composite_value_report`` at prod shape -> where does the wall go?
2. Grouped SSE reduction A/B: the shipped fused single-pass ``numba.njit`` kernel vs the ``np.bincount``
   fallback. The njit kernel computes counts + SSE(raw) + SSE(comp) + SSE(lag) in ONE sweep over the codes
   (no intermediate squared-error arrays); the bincount path does 3-4 O(n) passes. Measured ~9-10x on the
   isolated reduction (3.3 ms vs 32 ms), and folding the finite-mask into the same sweep drops the report
   end-to-end from ~127 ms to ~31 ms (4.1x). Verdict: njit is the DEFAULT (``_grouped_stats``), bincount is
   the no-numba fallback. After the reduction, ``pd.factorize`` (~16 ms) is the next cost.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite._value_report import build_composite_value_report

try:
    import numba

    @numba.njit(cache=True)
    def _grouped_stats_njit(codes, w, er, ec, el, n_groups):
        W = np.zeros(n_groups)
        sse_raw = np.zeros(n_groups)
        sse_comp = np.zeros(n_groups)
        sse_lag = np.zeros(n_groups)
        for i in range(codes.shape[0]):
            g = codes[i]
            wi = w[i]
            W[g] += wi
            sse_raw[g] += wi * er[i] * er[i]
            sse_comp[g] += wi * ec[i] * ec[i]
            sse_lag[g] += wi * el[i] * el[i]
        return W, sse_raw, sse_comp, sse_lag

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _grouped_stats_bincount(codes, w, er, ec, el, n_groups):
    W = np.bincount(codes, weights=w, minlength=n_groups)
    sse_raw = np.bincount(codes, weights=w * er * er, minlength=n_groups)
    sse_comp = np.bincount(codes, weights=w * ec * ec, minlength=n_groups)
    sse_lag = np.bincount(codes, weights=w * el * el, minlength=n_groups)
    return W, sse_raw, sse_comp, sse_lag


def _make_data(n=1_000_000, n_groups=773, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    y = rng.normal(50.0, 5.0, n)
    raw = y + rng.normal(0.0, 1.0, n)
    comp = y + rng.normal(0.0, 0.6, n)
    lag = y + rng.normal(0.0, 1.5, n)
    return y, raw, comp, lag, g


def _best_of(fn, *args, reps=7):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    y, raw, comp, lag, g = _make_data()
    n_groups = int(g.max()) + 1

    print("=== cProfile: build_composite_value_report (1M rows / 773 groups, with lag) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        build_composite_value_report(y, raw, comp, g, y_pred_lag=lag, expected_lift=0.3)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())

    t_full, _ = _best_of(lambda: build_composite_value_report(y, raw, comp, g, y_pred_lag=lag), reps=5)
    print(f"full report best-of-5: {1e3 * t_full:.2f} ms\n")

    print("=== grouped SSE reduction A/B (masked codes, n=1M) ===")
    codes = g.astype(np.int64)
    w = np.ones_like(y)
    er = raw - y
    ec = comp - y
    el = lag - y
    t_bc, _ = _best_of(_grouped_stats_bincount, codes, w, er, ec, el, n_groups)
    print(f"bincount (no-numba fallback): {1e3 * t_bc:.2f} ms")
    if _HAVE_NUMBA:
        _grouped_stats_njit(codes[:10], w[:10], er[:10], ec[:10], el[:10], n_groups)  # warm JIT
        t_nj, _ = _best_of(_grouped_stats_njit, codes, w, er, ec, el, n_groups)
        print(f"njit fused single-pass: {1e3 * t_nj:.2f} ms")
        print(f"njit speedup vs bincount: {t_bc / t_nj:.2f}x (>1 favours njit)")
    else:
        print("numba unavailable; bincount only")


if __name__ == "__main__":
    main()
