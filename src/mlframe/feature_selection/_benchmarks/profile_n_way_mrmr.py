"""cProfile audit of MRMR with ``interactions_max_order >= 3``.

Per CLAUDE.md "Profile every new feature with cProfile + optimize
hotspots": the n-way (3-way / 4-way) MRMR path was previously
untested and unprofiled. Combinatorial blowup (``C(p, k)``) means
order=3 generates ~C(p, 3) candidates per outer iteration and each
needs a KSG MI estimate. This profile measures whether order=3 has
actionable hotspots beyond the polynomial-from-pairs work already
optimised in earlier passes.

Run::

    python -m mlframe.feature_selection._benchmarks.profile_n_way_mrmr
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Wave 87 (2026-05-21): module-level filter removed; gated under __main__ below.

from mlframe.feature_selection.filters.mrmr import MRMR


def _make_3way_xor(n=2000, p=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int64)
    return X, y


def _wall(order, n=2000, p=10):
    X, y = _make_3way_xor(n=n, p=p)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    yser = pd.Series(y, name="y")
    t0 = time.perf_counter()
    sel = MRMR(interactions_max_order=order, verbose=0, random_seed=42)
    sel.fit(df, yser)
    return time.perf_counter() - t0, sel


def main():
    print("\n=== cProfile: MRMR n-way interactions, n=2000, p=10 ===\n")

    # Warmup.
    print("Warmup pass...", flush=True)
    _wall(1)

    # Wall-time scan over orders.
    print("\nWall-time scan (no profiler):")
    walls = {}
    for order in [1, 2, 3, 4]:
        wall, _ = _wall(order)
        walls[order] = wall
        print(f"  order={order}: {wall:6.2f}s")

    # Profile order=3 specifically (typical n-way regime).
    print("\nProfile order=3 with cProfile:")
    X, y = _make_3way_xor(n=2000, p=10)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    yser = pd.Series(y, name="y")
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    sel = MRMR(interactions_max_order=3, verbose=0, random_seed=42)
    sel.fit(df, yser)
    prof.disable()
    wall_with_prof = time.perf_counter() - t0
    overhead_pct = 100 * (wall_with_prof / walls[3] - 1)
    print(f"  with profiler: {wall_with_prof:.2f}s (overhead +{overhead_pct:.0f}%)")

    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumulative").print_stats(40)
    text = s.getvalue()
    print("\n--- Top 40 by cumulative time ---")
    print(text)

    # Also print top 30 by tottime (own time, not callees) -- this is
    # where actionable optimization opportunities live.
    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats("tottime").print_stats(30)
    print("\n--- Top 30 by own time (tottime) ---")
    print(s.getvalue())

    # Save to disk.
    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "profile_n_way_mrmr_order3.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# MRMR cProfile, interactions_max_order=3, n=2000, p=10\n")
        f.write(f"# wall (no profiler): {walls[3]:.2f}s\n")
        f.write(f"# wall (with profiler): {wall_with_prof:.2f}s\n")
        f.write(f"# profiler overhead: +{overhead_pct:.0f}%\n\n")
        full_s = io.StringIO()
        pstats.Stats(prof, stream=full_s).strip_dirs().sort_stats("cumulative").print_stats(80)
        f.write("--- Top 80 by cumulative time ---\n")
        f.write(full_s.getvalue())
        full_s = io.StringIO()
        pstats.Stats(prof, stream=full_s).strip_dirs().sort_stats("tottime").print_stats(50)
        f.write("\n--- Top 50 by own time ---\n")
        f.write(full_s.getvalue())
    print(f"\nFull profile -> {out_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
