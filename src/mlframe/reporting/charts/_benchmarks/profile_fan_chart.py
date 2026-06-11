"""cProfile harness for the FAN_CHART quantile panel at production shape.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_fan_chart``

FAN_CHART collapses the n-row forecast into ``_FAN_TIME_BUCKETS`` (400) equal-population horizon
buckets via bincount-based per-quantile means -- one O(n) pass per quantile column, no per-row python
loop. At n=1e6 / K=5 the whole panel is dominated by those K+2 bincount passes over n; there is no sort
and no length-n array retained in the returned spec (only the 400-bucket aggregates). Sub-second at 1e6.

Conclusion (see numbers printed below): the cost is K+2 vectorised bincount reductions over n. No
actionable speedup -- bincount is the right aggregate primitive and the work is already O(n) with a
small constant; njit/cuda would not beat numpy's C bincount at this call count (one panel per report).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.quantile import _fan_chart_panel


def _make_data(n: int):
    rng = np.random.default_rng(0)
    horizon = np.arange(n) / n
    center = np.sin(horizon * 6.0)
    half = 0.1 + 2.0 * horizon
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    z = (-1.6449, -0.6745, 0.0, 0.6745, 1.6449)
    preds = np.column_stack([center + zi * half for zi in z])
    y = center + rng.standard_normal(n) * half
    return y, preds, alphas


def _walltime(y, p, a, repeats: int = 3) -> float:
    _fan_chart_panel(y, p, a)  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _fan_chart_panel(y, p, a)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n in (100_000, 1_000_000, 5_000_000):
        y, p, a = _make_data(n)
        ms = _walltime(y, p, a) * 1e3
        print(f"n={n:>9}  fan_chart={ms:8.2f} ms")

    y, p, a = _make_data(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    _fan_chart_panel(y, p, a)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
