"""Bench: ewma_residual recurrence loop -- pure-Python scalar loop vs numba njit.

Target: stationarity.ewma_residual._ewma_single inner loop
    ewma[i] = alpha * x[i] + (1-alpha) * ewma[i-1]

This is a serial scalar recurrence over the full series, run once per half-life
per group. The pure-Python loop pays Python-frame overhead per element. An njit
kernel that performs the IDENTICAL float64 arithmetic in the IDENTICAL order is
bit-identical (same multiply/add sequence) and removes the interpreter overhead.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_ewma_residual_njit
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.stationarity import ewma_residual


def _old_ewma_single(seg: np.ndarray, hl: float, adjust: bool = False) -> np.ndarray:
    """Verbatim copy of the PRE-optimization _ewma_single (pure-Python loop)."""
    alpha = 1.0 - 2.0 ** (-1.0 / hl)
    seg_f = np.where(np.isfinite(seg), seg, 0.0)
    ewma = np.empty_like(seg_f)
    ewma[0] = seg_f[0]
    for i in range(1, seg_f.size):
        ewma[i] = alpha * seg_f[i] + (1.0 - alpha) * ewma[i - 1]
    if adjust:
        w = (1.0 - alpha) ** np.arange(seg_f.size)
        w_cum = np.cumsum(w[::-1])
        ewma = ewma * w[::-1] / w_cum
    return seg - ewma


def _bench(fn, *args, n_iter=20):
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for n in (2000, 50_000, 1_000_000):
        x = rng.standard_normal(n).cumsum()
        hl_list = [5.0, 20.0, 60.0, 240.0]

        # identity: full ewma_residual (new path) vs old python-loop reconstruction.
        new = ewma_residual(x, half_life=hl_list)
        old = np.full((n, len(hl_list)), np.nan)
        for j, hl in enumerate(hl_list):
            old[:, j] = _old_ewma_single(x, hl)
        ident = np.array_equal(new, old, equal_nan=True)
        max_abs = np.nanmax(np.abs(new - old)) if not ident else 0.0

        # warm njit
        ewma_residual(x[:100], half_life=hl_list)

        t_old = _bench(lambda: [_old_ewma_single(x, hl) for hl in hl_list])
        t_new = _bench(lambda: ewma_residual(x, half_life=hl_list))
        print(f"n={n:>9} OLD={t_old*1e3:9.3f}ms NEW={t_new*1e3:9.3f}ms " f"speedup={t_old/t_new:5.2f}x identical={ident} max_abs={max_abs:.2e}")


if __name__ == "__main__":
    main()
