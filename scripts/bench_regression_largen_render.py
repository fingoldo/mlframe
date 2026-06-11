"""Profile the large-n regression pred-vs-actual hexbin/log-density spec-build path.

Run: python scripts/bench_regression_largen_render.py [--n 1000000 5000000 10000000] [--profile]

Measures wall time of the SCATTER (hexbin/log-density) panel and the full default 4-panel template at very large n
(the user's "slow on big data" concern). The expensive work is spec-construction (the O(n) binning / residual passes),
NOT the matplotlib draw -- above the hexbin threshold the cloud is a fixed density_bins x density_bins matrix, so the
renderer cost is n-independent. With --profile, dumps a cProfile cumulative-time breakdown so the hotspot (histogram2d
binning, the per-panel _finite_pair float64 copy, residual hist, decile quantile) is attributable.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.regression import (
    DEFAULT_REGRESSION_PANELS, _scatter_panel, compose_regression_figure,
)
from mlframe.training.targets import audit_residuals


def make_data(n: int, seed: int = 20260611):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=n)
    yt = 2.0 * x + 5.0 + rng.normal(0.0, 1.0, size=n)
    yp = 2.0 * x + 5.0 + rng.normal(0.0, 0.7, size=n)
    return yt, yp


def time_call(fn, repeats: int = 3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1_000_000, 5_000_000, 10_000_000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="dump cProfile breakdown at the largest n")
    args = ap.parse_args()

    print(f"{'n':>12} {'SCATTER(hexbin) ms':>20} {'full 4-panel ms':>18}")
    for n in args.n:
        yt, yp = make_data(n)
        audit = audit_residuals(yt, yp)
        scatter_ms = time_call(lambda: _scatter_panel(yt, yp, title="m"), args.repeats) * 1e3
        full_ms = time_call(
            lambda: compose_regression_figure(yt, yp, audit=audit, panels_template=DEFAULT_REGRESSION_PANELS),
            args.repeats,
        ) * 1e3
        print(f"{n:>12_} {scatter_ms:>20.1f} {full_ms:>18.1f}")

    if args.profile:
        n = max(args.n)
        yt, yp = make_data(n)
        audit = audit_residuals(yt, yp)
        for label, fn in (
            ("SCATTER hexbin", lambda: _scatter_panel(yt, yp, title="m")),
            ("full 4-panel", lambda: compose_regression_figure(yt, yp, audit=audit, panels_template=DEFAULT_REGRESSION_PANELS)),
        ):
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(args.repeats):
                fn()
            pr.disable()
            s = StringIO()
            pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(25)
            print(f"\n===== cProfile cumulative: {label} @ n={n:_} ({args.repeats}x) =====")
            print(s.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
