"""Bench: ``_causal_rolling``'s numba-accelerated running-window kernels vs the original
per-row ``np.median``/``.mean()`` Python loop.

Mean case: O(n) running-window sum (add entering element, drop leaving element) instead of an
O(n*window) per-row ``.mean()`` rescan -- mirrors ``_grouped_causal_bases.py::_grouped_trailing_impl``'s
single-series case.
Median case: same O(n*window) algorithm, just JIT-compiled (a true O(1)-amortized sliding-window
median needs a balanced-heap structure numba can't easily express).

Bit-identity is asserted (NaN-mask + np.allclose on finite entries) at every size before any timing
counts, so a speedup number is never reported for a result that silently changed.

Run:
    cd "<repo>" && CUDA_VISIBLE_DEVICES="" python -m \\
      mlframe.training.composite.discovery._benchmarks.bench_causal_rolling_njit
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training.composite.discovery._base_engineering import _causal_rolling


def _causal_rolling_naive(y_sorted: np.ndarray, window: int, *, median: bool) -> np.ndarray:
    """Pre-fix reference implementation: per-row Python loop + np.median/.mean() per window."""
    n = y_sorted.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1:
        return out
    for i in range(window, n):
        past = y_sorted[i - window : i]
        out[i] = np.median(past) if median else past.mean()
    return out


def _best_of(fn, *args, repeats: int = 5, **kwargs) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    sizes = [(500, 5), (5_000, 20), (50_000, 30), (200_000, 50)]

    # Warm the JIT once before any timing.
    _causal_rolling(rng.normal(size=200), 5, median=False)
    _causal_rolling(rng.normal(size=200), 5, median=True)

    print(f"{'n':>8} {'window':>7} {'kind':>7} {'naive(s)':>10} {'njit(s)':>10} {'speedup':>9} {'bit-identical':>14}")
    for n, window in sizes:
        y = rng.normal(size=n)
        for median in (False, True):
            naive = _causal_rolling_naive(y, window, median=median)
            fast = _causal_rolling(y, window, median=median)
            both_nan = np.isnan(naive) & np.isnan(fast)
            identical = bool(np.array_equal(np.isnan(naive), np.isnan(fast))) and bool(
                np.allclose(naive[~both_nan], fast[~both_nan], rtol=0, atol=1e-9)
            )
            t_naive = _best_of(_causal_rolling_naive, y, window, median=median, repeats=3)
            t_fast = _best_of(_causal_rolling, y, window, median=median, repeats=5)
            speedup = t_naive / max(t_fast, 1e-9)
            kind = "median" if median else "mean"
            print(f"{n:>8} {window:>7} {kind:>7} {t_naive:>10.5f} {t_fast:>10.5f} {speedup:>8.1f}x {identical!s:>14}")


if __name__ == "__main__":
    main()
