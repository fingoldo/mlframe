"""cProfile + njit-vs-numpy A/B harness for the category-discriminability WoE screen at production shape.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_category_discriminability``

The hot kernel is ``level_woe``'s per-row count pass (positive-count + total-count per level). This benches the njit
``_level_counts_njit`` against the two-``np.bincount`` numpy fallback at n in {100k, 1M} with ~50 levels, then cProfiles the
full ``category_discriminability_table`` at n=1M. Numbers are pasted into the module docstring verdict.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

import mlframe.reporting.charts.category_discriminability as cd
from mlframe.reporting.charts.category_discriminability import category_discriminability_table, level_woe


def _make(n: int, n_levels: int = 50):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, n_levels, size=n).astype(np.int64)
    rates = np.linspace(0.2, 0.8, n_levels)
    y = (rng.random(n) < rates[codes]).astype(np.float64)
    return codes, y


def _numpy_counts(codes, y, n_levels):
    keep = codes >= 0
    kc = codes[keep]
    tot = np.bincount(kc, minlength=n_levels).astype(np.float64)
    pos = np.bincount(kc, weights=y[keep], minlength=n_levels)
    return pos, tot


def _best(fn, *args, repeats: int = 5) -> float:
    fn(*args)  # warm (numba compile / cache fill)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n_levels = 50
    print(f"has_numba={cd._HAS_NUMBA}  n_levels={n_levels}")
    for n in (100_000, 1_000_000):
        codes, y = _make(n, n_levels)
        njit_ms = _best(cd._level_counts_njit, codes, y, n_levels) * 1e3
        np_ms = _best(_numpy_counts, codes, y, n_levels) * 1e3
        speedup = np_ms / njit_ms if njit_ms else float("nan")
        print(f"n={n:>9}  count-pass njit={njit_ms:8.3f} ms  numpy_bincount={np_ms:8.3f} ms  njit_speedup={speedup:5.2f}x")

    n = 1_000_000
    codes, y = _make(n, n_levels)
    labels = [f"L{i}" for i in range(n_levels)]
    X = pd.DataFrame({"f": pd.Categorical.from_codes(codes, categories=labels)})
    level_woe(codes, y, n_levels, float(y.mean()))  # warm
    pr = cProfile.Profile()
    pr.enable()
    category_discriminability_table(X, y, top_k=15)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
