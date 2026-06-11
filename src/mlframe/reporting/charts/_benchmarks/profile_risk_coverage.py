"""cProfile harness for the risk-coverage curve at production scale.

The curve is argsort-dominated by construction: one O(n log n) descending confidence sort, then one O(n) cumulative
pass. At n=1e6 the sort is the only meaningful cost; there is no per-row Python loop to optimise. Run:

    python -m mlframe.reporting.charts._benchmarks.profile_risk_coverage
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.reporting.charts.risk_coverage import build_risk_coverage_spec, compute_risk_coverage


def _data(n: int):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    conf = rng.uniform(0.0, 1.0, n)
    correct = rng.uniform(0, 1, n) < (0.45 + 0.54 * conf)
    pred = np.where(correct, y, 1 - y)
    score = np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
    return y, score


def main() -> None:
    n = 1_000_000
    y, score = _data(n)
    compute_risk_coverage(y[:1000], score[:1000], task="binary")  # warm

    t0 = time.perf_counter()
    res = build_risk_coverage_spec(y, score, task="binary")
    wall = time.perf_counter() - t0
    print(f"n={n}  wall={wall * 1e3:.1f} ms  AURC={res.aurc:.4f}  plotted_points={res.figure.panels[0][0].x.size}")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        build_risk_coverage_spec(y, score, task="binary")
    pr.disable()
    stats = pstats.Stats(pr).sort_stats("cumulative")
    stats.print_stats(15)


if __name__ == "__main__":
    main()
