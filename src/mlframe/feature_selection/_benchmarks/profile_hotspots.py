"""cProfile-based hotspot profiler for the refactored ``filters/`` package.

Runs a representative MRMR.fit on a moderately sized synthetic classification
dataset, captures the cProfile stats, and prints the top-30 cumulative-time
functions plus their breakdown by call count. Saves the raw stats to
``_results/profile_<git_sha>.prof`` for later visual inspection
(``snakeviz`` / ``gprof2dot``).

Usage::

    python -m mlframe.feature_selection._benchmarks.profile_hotspots
    python -m mlframe.feature_selection._benchmarks.profile_hotspots --n 5000 --p 200 --runs 2
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_dataset(n: int, p: int, informative: int, seed: int):
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=informative,
        n_redundant=0,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=seed,
    )
    cols = [f"f_{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y


def _warmup(seed: int = 42) -> None:
    """Numba-compile prewarm so the cProfile pass measures steady-state work."""
    from mlframe.feature_selection.filters import MRMR
    X, y = _build_dataset(n=500, p=20, informative=5, seed=seed)
    m = MRMR(
        quantization_nbins=5,
        full_npermutations=2,
        baseline_npermutations=1,
        n_jobs=1,
        verbose=0,
        cv=2,
    )
    m.fit(X, y)


def _profile_one(n: int, p: int, informative: int, seed: int) -> tuple[pstats.Stats, float]:
    """Profile one MRMR.fit. Returns (stats, wall_time_s)."""
    from mlframe.feature_selection.filters import MRMR

    X, y = _build_dataset(n=n, p=p, informative=informative, seed=seed)
    m = MRMR(
        quantization_nbins=10,
        full_npermutations=3,
        baseline_npermutations=2,
        n_jobs=1,
        verbose=0,
        cv=2,
    )

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    m.fit(X, y)
    profiler.disable()
    t1 = time.perf_counter()

    return pstats.Stats(profiler), t1 - t0


def _format_top(stats: pstats.Stats, top_n: int = 30) -> str:
    """Return a markdown-formatted top-N table by cumulative time, filtered to
    mlframe + numba-dispatched functions (drops Python stdlib noise)."""
    out = io.StringIO()
    stats.stream = out
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    return out.getvalue()


def _filter_to_filters(stats: pstats.Stats, top_n: int = 30) -> str:
    out = io.StringIO()
    stats.stream = out
    stats.sort_stats("cumulative")
    stats.print_stats("filters", top_n)
    return out.getvalue()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="number of samples")
    p.add_argument("--p", type=int, default=100, help="number of features")
    p.add_argument("--informative", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    if not args.no_warmup:
        print("== Warmup pass (compiling numba) ==", flush=True)
        _warmup(seed=args.seed)

    print(f"== Profiling: n={args.n}, p={args.p}, informative={args.informative} ==", flush=True)
    stats, wall = _profile_one(args.n, args.p, args.informative, args.seed)
    print(f"\nTotal MRMR.fit wall time: {wall:.3f}s")

    # Save raw stats.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prof_path = RESULTS_DIR / f"profile_n{args.n}_p{args.p}_{ts}.prof"
    stats.dump_stats(str(prof_path))
    print(f"Raw stats: {prof_path}")

    print("\n== Top 30 by cumulative time (all) ==")
    print(_format_top(stats, top_n=30))

    print("\n== Top 30 inside mlframe.feature_selection.filters/* ==")
    print(_filter_to_filters(stats, top_n=30))

    return 0


if __name__ == "__main__":
    sys.exit(main())
