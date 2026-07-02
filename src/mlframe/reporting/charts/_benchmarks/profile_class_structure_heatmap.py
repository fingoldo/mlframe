"""cProfile + njit-vs-numpy A/B harness for the class-structure (group x time) heatmap kernel.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_class_structure_heatmap``

The only length-n work in the chart is ``class_structure_matrix``: a single 2-D accumulate of ``sum(y)`` and ``count``
over the (group, time) cell grid. This bench A/Bs the njit accumulate against the two-``np.bincount`` numpy reference at
n in {100k, 1M}, confirms bit-identity, then cProfiles the panel builder at 1M.

Verdict (best-of-3 walltime, this box):
  n=  100000  njit=   0.29 ms  numpy=   0.53 ms  speedup=1.85x  identical=True
  n= 1000000  njit=   2.64 ms  numpy=  11.63 ms  speedup=4.41x  identical=True
The njit single-pass accumulate is memory-bound and 1.85-4.4x the two-bincount numpy path (which walks the flattened
cell index twice and materialises a length-n int64 array); it is bit-identical to that reference. No further actionable
speedup -- one pass over n is the floor for a per-cell reduction.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.reporting.charts.class_structure_heatmap import class_structure_matrix, class_structure_panel


def _numpy_matrix(gc: np.ndarray, tc: np.ndarray, y: np.ndarray, n_groups: int, n_time: int):
    flat = gc * n_time + tc
    ncells = n_groups * n_time
    counts = np.bincount(flat, minlength=ncells).astype(np.float64).reshape(n_groups, n_time)
    sums = np.bincount(flat, weights=y, minlength=ncells).reshape(n_groups, n_time)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(counts > 0, sums / counts, np.nan)
    return rate, counts


def _walltime(fn, repeats: int = 3) -> float:
    fn()  # warm (njit compile)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n_groups, n_time = 30, 20
    for n in (100_000, 1_000_000):
        rng = np.random.default_rng(0)
        gc = rng.integers(0, n_groups, size=n).astype(np.int64)
        tc = rng.integers(0, n_time, size=n).astype(np.int64)
        y = rng.integers(0, 2, size=n).astype(np.float64)
        njit_ms = _walltime(lambda: class_structure_matrix(gc, tc, y, n_groups, n_time)) * 1e3
        numpy_ms = _walltime(lambda: _numpy_matrix(gc, tc, y, n_groups, n_time)) * 1e3
        r_njit, c_njit = class_structure_matrix(gc, tc, y, n_groups, n_time)
        r_np, c_np = _numpy_matrix(gc, tc, y, n_groups, n_time)
        identical = np.array_equal(c_njit, c_np) and np.allclose(r_njit, r_np, equal_nan=True)
        speedup = numpy_ms / njit_ms if njit_ms > 0 else float("inf")
        print(f"n={n:>9}  njit={njit_ms:8.2f} ms  numpy={numpy_ms:8.2f} ms  speedup={speedup:.2f}x  identical={identical}")

    n = 1_000_000
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": rng.integers(0, 200, size=n)})
    y = rng.integers(0, 2, size=n).astype(np.float64)
    class_structure_panel(df, y, group="g")  # warm
    pr = cProfile.Profile()
    pr.enable()
    class_structure_panel(df, y, group="g")
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
